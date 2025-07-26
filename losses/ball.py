import tensorflow as tf

tf.keras.backend.set_floatx("float64")
from geometry.common import compute_ricci_tensor
from helper_functions.helper_functions import RadiusWeighting, cholesky_from_vec
from geometry.common import compute_scalar_curvature
from geometry.ball import prescribed_K, patch_xy_to_xyz, PatchChange_Coordinates_Ball, PatchChange_Metric_Ball

class ScalarLoss:
    """
    Computes the loss |R(g) - K_prescribed|^2 for a given patch.
    This class is self-contained and handles the geometry calculation.
    """
    def __init__(self, hp, patch_idx=0):
        self.hp = hp
        self.K_kind = self.hp.get("K_kind", "round")
        self.patch_idx = int(patch_idx)

    def compute(self, coords, patch_model_callable):
        """
        Args:
            coords (tf.Tensor): The coordinates for this patch.
            metric_cholesky_vec_callable (callable): A function that takes coordinates 
                                                      and returns the predicted Cholesky vector.
        """
        # compute_scalar_curvature expects a model/callable that it can pass coordinates to.
        scalar_curv_pred = compute_scalar_curvature(coords, patch_model_callable)
        K_target = prescribed_K(coords, kind=self.K_kind, patch_idx=self.patch_idx)
        
        return tf.reduce_mean(tf.square(scalar_curv_pred - K_target))


class ConformalLoss:
    """
    Computes the total loss for the GlobalConformalModel_L2.
    This version correctly uses the model's patch callables.
    """
    def __init__(self, hp, print_losses=False):
        self.hp = hp
        self.n_patches = self.hp.get("n_patches", 2)

        self.print_losses = print_losses
        self.print_interval = self.hp.get("print_interval", 100) # Print every 100 steps
        self.step_count = tf.Variable(0, dtype=tf.int64, trainable=False, name="conformal_loss_step_counter")

        self.scalar_loss_calculators = [
            ScalarLoss(hp=self.hp, patch_idx=i) for i in range(self.n_patches)
        ]
        
        self.scalar_loss_multiplier = self.hp.get("scalar_loss_multiplier", 1.0)
        self.finiteness_multiplier = self.hp.get("finiteness_multiplier", 0.01)

    def call(self, model, x_vars, metric_pred, return_constituents=False, val_print=True):
        """
        Calculates the total loss.
        Note: The `metric_pred` argument is no longer used here, as we get the
              predictions by calling the submodels directly.
        """
        # 1. Get the patch coordinates and the corresponding model callables.
        if self.n_patches == 2:
            coords_p2 = model.patch_transform_layer(x_vars)
            
            # The GlobalConformalModel_L2 conveniently provides these callables.
            callable_p1 = model.patch_submodels[0]
            callable_p2 = model.patch_submodels[1]

            # 2. Compute the scalar loss for each patch.
            loss1 = self.scalar_loss_calculators[0].compute(x_vars, callable_p1)
            loss2 = self.scalar_loss_calculators[1].compute(coords_p2, callable_p2)
            scalar_losses = [loss1, loss2]

        else: # Single patch case
            # For a single patch, the main model itself is the callable.
            loss1 = self.scalar_loss_calculators[0].compute(x_vars, model)
            scalar_losses = [loss1]

        total_scalar_loss = tf.add_n(scalar_losses)

        # 3. Finiteness/Regularization Loss on the conformal factor `u`.
        xyz_coords = patch_xy_to_xyz(x_vars, patch_idx=0)
        u_vals = model.u_model(xyz_coords)
        finiteness_loss_val = tf.reduce_mean(tf.square(u_vals))

        # 4. Combine the losses.
        total_loss = (self.scalar_loss_multiplier * total_scalar_loss + 
                      self.finiteness_multiplier * finiteness_loss_val)
        
        if self.print_losses:
            self.step_count.assign_add(1)
            if tf.equal(self.step_count % self.print_interval, 0):
                tf.print("Total Combined Loss:", total_loss)
                for i, s_loss in enumerate(scalar_losses):
                    tf.print("Scalar Loss Patch", i, ":", s_loss)
                tf.print("Finiteness Loss (u^2):", finiteness_loss_val)

        # 5. Handle logging.
        if return_constituents:
            loss_constituents = {
                "scalar_loss": total_scalar_loss,
                "finiteness_loss": finiteness_loss_val,
                "total_loss": total_loss,
            }
            for i, loss in enumerate(scalar_losses):
                loss_constituents[f"scalar_loss_patch_{i}"] = loss
            return total_loss, loss_constituents
        
        return total_loss, None
    
class TotalBallLoss:
    """
    Represents a class for computing the total training loss, which has
    contributions from solving the Einstein equation, from satisfying overlap
    conditions of patches, and from finiteness of the metric components.

    Attributes:
    - hp (dict): Dictionary of the training hyperparameters.
    - num_dimensions (int): Number of dimensions of the manifold.
    - n_patches (int): Number of patches in the manifold definition (1 or 2).
    - overlap_upperwidth (float): distance from the radial midpoint to the edge
      of the patch area of interest, beyond here the radial filter devalues point contributions.
    - print_losses (bool): Whether to print batch loss values with each call.
    - einstein_constant (float): The proportioanlity constant in the Einstein equation.
    - einstein_multiplier (float): The weighted contribution of the Einstein losses to the total loss.
    - overlap_multiplier (float): The weighted contribution of the overlap losses to the total loss.
    - finiteness_multiplier (float): The weighted contribution of the finiteness losses to the total loss.
    - einstein_losses (list): The Einstein loss values across the patches.
    - overlap_loss (flaot): The overlap loss values for the model.
    - filter_hyperparameters (list): The defining parameters for the finiteness filter function.
    - finite_losses (list): The Finiteness loss values across the patches.

    Methods:
    - __init__(self, hp, print_losses):
      Initializes the TotalLoss class with the respective loss hyperparameters,
      initialising the loss components and multipliers.

    - call(self, x_vars, return_constituents, val_print):
      Computes the total loss, which is defined as a weighted sum of the loss
      components: Einstein, Overlap, Finiteness.
    """

    def __init__(self, hp, print_losses=False):
        self.hp = hp
        self.num_dimensions = self.hp["dim"]
        self.n_patches = self.hp["n_patches"]
        self.overlap_upperwidth = self.hp["overlap_upperwidth"]
        self.print_losses = print_losses
        self.print_interval = self.hp["print_interval"]
        self.step_count = 0

        # Set the Loss type
        self.loss_type = self.hp.get("loss_type", "einstein")

        # Initialize the scalar loss if required
        if self.loss_type == "scalar":
            if self.n_patches == 1:
                self.scalar_losses = [
                    ScalarLoss(kind=self.hp.get("K_kind", "zero"),patch_idx=0)
                ]
            else:
                self.scalar_losses = [
                    ScalarLoss(kind=self.hp.get("K_kind", "zero"), patch_idx=i)
                    for i in range(int(self.n_patches))
                ]

        # Initialize Einstein loss if needed
        elif self.loss_type == "einstein":
            self.einstein_constant = self.hp["einstein_constant"]
            if self.n_patches == 1:
                self.einstein_losses = [
                    EinsteinLoss(self.num_dimensions, self.einstein_constant, False)
                ]
            else:
                self.einstein_losses = [
                    EinsteinLoss(
                        self.num_dimensions,
                        self.einstein_constant,
                        True,
                        self.overlap_upperwidth,
                    )
                    for _ in range(int(self.n_patches))
                ]


        # Loss multipliers
        self.einstein_multiplier = self.hp["einstein_multiplier"]
        self.overlap_multiplier = self.hp["overlap_multiplier"]
        self.finiteness_multiplier = self.hp["finiteness_multiplier"]
        assert (
            abs(self.einstein_multiplier)
            + abs(self.overlap_multiplier)
            + abs(self.finiteness_multiplier)
            > 0.0
        ), "All loss terms turned off..."
        if self.n_patches == 1:
            self.overlap_multiplier = tf.cast(0.0, tf.float64)

        # Finiteness Loss
        self.filter_hyperparameters = [
            self.hp["finite_centre"],
            self.hp["finite_width"],
            self.hp["finite_sharpness"],
            self.hp["finite_height"],
            self.hp["finite_slope"],
        ]
        self.finite_losses = [
            FiniteLoss(self.num_dimensions, self.filter_hyperparameters)
            for patch in range(int(self.n_patches))
        ]

    def call(
        self, model, x_vars, metric_pred, return_constituents=False, val_print=True
    ):
        # Set up the network inputs & outputs
        patch_inputs = [x_vars]
        metric_preds = []
        if self.n_patches > 1:
            # Compute the input coordinates in the second patch
            patch_inputs.append(model.patch_transform_layer(x_vars))

            # Split the output into the metrics in each patch
            patch_1_output, patch_2_output = tf.split(
                metric_pred, num_or_size_splits=2, axis=-1
            )
            metric_preds.append(patch_1_output)
            metric_preds.append(patch_2_output)
        else:
            metric_preds.append(metric_pred)

        # Compute the loss components
        if self.loss_type == "scalar":
            e_losses = [
                self.scalar_losses[i].compute(
                    patch_inputs[i], model.patch_submodels[i]
                )
                for i in range(self.n_patches)
            ]
        elif self.loss_type == "einstein":
            e_losses = [
                self.einstein_losses[i].compute(
                    patch_inputs[i], metric_preds[i], model.patch_submodels[i]
                )
                for i in range(self.n_patches)
            ]
        else:
            raise ValueError(f"Unsupported loss_type: {self.loss_type}")

        # Overlap
        if self.overlap_multiplier > 0.0 and self.n_patches == 2:
            overlap_loss = self.overlap_loss.compute(
                x_vars, [metric_preds[0], metric_preds[1]]
            )
        else:
            overlap_loss = tf.cast(0.0, tf.float64)

        # Finiteness
        if self.finiteness_multiplier > 0.0:
            f_losses = [
                tf.math.log(
                    self.finite_losses[patch_idx].compute(metric_preds[patch_idx])
                )
                for patch_idx in range(int(self.n_patches))
            ]
        else:
            f_losses = [
                tf.cast(0.0, tf.float64) for _ in range(int(self.n_patches))
            ]

        # Print the batch loss values
        if self.print_losses and (self.step_count + 1) % self.print_interval == 0:
            print(
                f"Einstein: {[tf.get_static_value(e_loss) for e_loss in e_losses]}\nOverlap: {tf.get_static_value(overlap_loss)}\nFinite: {[tf.get_static_value(f_loss) for f_loss in f_losses]}\n"
            )

        # Initialise the constituent losses dictionary (holds each of the
        # loss components pre-sum)
        if return_constituents:
            loss_constituents = {
                "einstein_losses": [tf.get_static_value(e_loss) for e_loss in e_losses],
                "overlap_loss": tf.get_static_value(overlap_loss),
                "finiteness_losses": [
                    tf.get_static_value(f_loss) for f_loss in f_losses
                ],
            }
        else:
            loss_constituents = None

        # Compute the total loss (accounting for multipliers)
        total_loss = 0.0
        if self.einstein_multiplier > 0.0:
            total_loss += self.einstein_multiplier * tf.reduce_sum(
                tf.math.abs(e_losses)
            )
        if self.overlap_multiplier > 0.0:
            total_loss += self.overlap_multiplier * tf.math.abs(overlap_loss)
        if self.finiteness_multiplier > 0.0:
            total_loss += self.finiteness_multiplier * tf.reduce_sum(
                tf.math.abs(f_losses)
            )
        # Normalise by the multiplier factors
        total_loss /= (
            self.einstein_multiplier
            + self.overlap_multiplier
            + self.finiteness_multiplier
        )

        self.step_count += 1
        return total_loss, loss_constituents


class EinsteinLoss:
    """
    Represents a class for computing the Einstein loss, which measures the
    difference between the Ricci tensor and the predicted metric tensor
    (scaled by the Einstein constant $lambda$).

    Attributes:
    - dim (int): The dimensionality of the metric tensor (number of x coords).
    - einstein_constant (float): The proportioanlity constant in the Einstein equation.
    - weight_radially (bool): Whether to apply radial weighting to the points.
    - overlap_upperwidth (float): distance from the radial midpoint to the edge
      of the patch area of interest, beyond here the radial filter devalues point contributions.

    Methods:
    - __init__(self, num_dimensions, einstein_constant, weight_radially, overlap_upperwidth):
      Initializes the EinsteinLoss class with the respective loss hyperparameters.

    - compute(self, x_vars, metric_pred, model):
      Computes the Einstein loss, which is defined as the squared norm of the
      difference between the Ricci tensor and the predicted metric tensor
      (scaled by a constant).
    """

    def __init__(
        self,
        num_dimensions,
        einstein_constant=1.0,
        weight_radially=True,
        overlap_upperwidth=0.1,
    ) -> None:
        self.dim = num_dimensions
        self.einstein_constant = einstein_constant
        self.weight_radially = weight_radially
        self.overlap_upperwidth = overlap_upperwidth

    def compute(self, x_vars, metric_pred, model):
        # Compute the Ricci tensor
        ricci_tensor = compute_ricci_tensor(x_vars, model)
        # Convert the metric vielbein to a matrix
        metric_pred_mat = cholesky_from_vec(metric_pred)
        # Compute the loss from the Einstein equation
        norm = tf.norm(
            self.einstein_constant * metric_pred_mat - ricci_tensor, axis=(1, 2)
        )

        # Apply radial weighting
        if self.weight_radially:
            radial_midpoint = tf.cast(tf.sqrt(2.0) - 1.0, tf.float64)
            filter_width = radial_midpoint + self.overlap_upperwidth
            radial_weights = RadiusWeighting(x_vars, filter_width)
            norm = norm * radial_weights

        einstein_loss = tf.reduce_mean(norm)

        return einstein_loss


class FiniteLoss:
    """
    Represents a class for computing the finiteness loss, which measures the
    norm of the metric components and weights according to a predefined filter.
    This loss component ensures the zero metric is avoided as an attractor point
    of the learning.

    Attributes:
    - dim (int): The dimensionality of the metric tensor (number of x coords).
    - filter_hyperparameters (list): The defining parameters for the finiteness filter function.

    Methods:
    - __init__(self, num_dimensions, filter_hyperparameters):
      Initializes the FinitenessLoss class with the respective loss hyperparameters.

    - compute(self, metric_pred):
      Computes the Finiteness loss, which is defined as the norm of the
      metric components after applying a filter weighting function.
    """

    def __init__(self, num_dimensions, filter_hyperparameters) -> None:
        self.dim = num_dimensions
        self.filter_hyperparameters = filter_hyperparameters

    def compute(self, metric_pred):
        # Convert the metric vielbein to a matrix
        metric_pred_mat = cholesky_from_vec(metric_pred)

        # Import filter hyperparameters
        finite_centre = self.filter_hyperparameters[0]
        finite_width = self.filter_hyperparameters[1]
        finite_sharpness = self.filter_hyperparameters[2]
        finite_height = self.filter_hyperparameters[3]
        finite_slope = self.filter_hyperparameters[4]

        # Compute the norm of the metric components
        sum_metric_pred = (
            tf.reduce_sum(abs(metric_pred_mat), axis=[1, 2], keepdims=True)
            * 2
            / ((self.dim) * (self.dim - 1))
        )

        # Define the finiteness filter weighting function
        gaussian_weight = (
            (
                tf.square(
                    finite_height
                    * tf.exp(
                        -tf.pow(
                            ((sum_metric_pred - finite_centre) / finite_width),
                            finite_sharpness,
                        )
                    )
                    - finite_height
                )
                + 1
            )
            + (
                sum_metric_pred / finite_slope
                - (finite_centre + finite_width) / finite_slope
            )
            * (
                1
                + tf.math.tanh(sum_metric_pred / 2 - (finite_centre + finite_width) / 2)
            )
            / 2
            + (
                -sum_metric_pred / finite_slope
                + (finite_centre - finite_width) / finite_slope
            )
            * (
                1
                + tf.math.tanh(
                    -sum_metric_pred / 2 + (finite_centre - finite_width) / 2
                )
            )
            / 2
        )

        finite_loss = tf.square(1 - tf.reduce_mean(gaussian_weight)) + 1

        return finite_loss


class GlobalLossBall:
    """
    Represents a class for computing the global test loss, which has
    contributions from solving the Einstein equation and from satisfying overlap
    conditions of patches. The patches are restricted to points within the radial
    limit, and the overlap region is an annulus which spans either side of the
    radial midpoint and runs up to the radial limit, such that it is symmetric
    under the patch transform function.

    Attributes:
    - hp (dict): Dictionary of the training hyperparameters.
    - num_dimensions (int): Number of dimensions of the manifold.
    - n_patches (int): Number of patches in the manifold definition (1 or 2).
    - radial_limit (float): The hard radial boundary for the patches.
    - radial_midpoint (float): The radial value which maps to itself under the function which transofrms between patches.
    - einstein_constant (float): The proportioanlity constant in the Einstein equation.
    - einstein_multiplier (float): The weighted contribution of the Einstein losses to the total loss.
    - overlap_multiplier (float): The weighted contribution of the overlap losses to the total loss.
    - einstein_losses (list): The Einstein loss values across the patches.
    - overlap_loss (float): The overlap loss values for the model.

    Methods:
    - __init__(self, hp, radial_limit):
      Initializes the GlobalLossBall class with the respective loss hyperparameters,
      initialising the loss components and multipliers.

    - call(self, model, x_vars, metric_pred):
      Computes the global test loss, which is defined as a weighted sum of the loss
      components: Einstein, Overlap.
    """

    def __init__(self, hp, radial_limit=None):
        self.hp = hp
        self.num_dimensions = self.hp["dim"]
        self.n_patches = self.hp["n_patches"]
        self.radial_limit = radial_limit
        self.radial_midpoint = tf.cast(tf.sqrt(2.0) - 1.0, tf.float64)
        # Ensure the patching conditions are consistently defined
        if self.radial_limit:
            assert self.radial_limit > self.radial_midpoint, "Patches do not overlap..."

        # Einstein constant, $\lambda$ in the Einstein equation: $R_{ij} = \lambda g_{ij}$
        self.einstein_constant = self.hp["einstein_constant"]

        # Loss multipliers
        self.einstein_multiplier = self.hp["einstein_multiplier"]
        self.overlap_multiplier = self.hp["overlap_multiplier"]
        if self.n_patches == 1:
            self.overlap_multiplier = tf.cast(0.0, tf.float64)

        # Einstein Loss
        self.einstein_losses = [
            EinsteinLoss(
                self.num_dimensions, self.einstein_constant, weight_radially=False
            )
            for _ in range(int(self.n_patches))
        ]


    def call(self, model, x_vars, metric_pred):
        # Set up the network inputs & outputs
        patch_inputs = [x_vars]
        metric_preds = []
        if self.n_patches > 1:
            # Compute the input coordinates in the second patch
            patch_inputs.append(model.patch_transform_layer(x_vars))

            # Split the output into the metrics in each patch
            patch_1_output, patch_2_output = tf.split(
                metric_pred, num_or_size_splits=2, axis=-1
            )
            metric_preds.append(patch_1_output)
            metric_preds.append(patch_2_output)
        else:
            metric_preds.append(metric_pred)

        # Compute data limited to each patch
        if self.radial_limit and self.radial_limit > 0:
            # Patches
            norms = [
                tf.sqrt(tf.reduce_sum(tf.square(p_pts), axis=1))
                for p_pts in patch_inputs
            ]
            masks = [
                norm < self.radial_limit for norm in norms
            ]  # ...find points within the radial limit
            pts_limited = [
                tf.boolean_mask(patch_inputs[p_idx], masks[p_idx])
                for p_idx in range(int(self.n_patches))
            ]
            metrics_limited = [
                tf.boolean_mask(metric_preds[p_idx], masks[p_idx])
                for p_idx in range(int(self.n_patches))
            ]

            # Overlap Region
            mask_overlap = tf.logical_and(
                norms[0] >= (1 - self.radial_limit) / (1 + self.radial_limit),
                norms[0] <= self.radial_limit,
            )  # ...find points within the overlap region
            pts_overlap = tf.boolean_mask(patch_inputs[0], mask_overlap)
            metrics_overlap = [
                tf.boolean_mask(metric_preds[p_idx], mask_overlap)
                for p_idx in range(int(self.n_patches))
            ]
        else:
            # ...otherwise use the full patches in each case
            pts_limited, metrics_limited, pts_overlap, metrics_overlap = (
                patch_inputs,
                metric_preds,
                patch_inputs[0],
                metric_preds,
            )

        # Compute the number of points in each region
        sample_sizes = [[p_pts.shape[0] for p_pts in pts_limited], pts_overlap.shape[0]]

        # Compute the loss components
        if self.einstein_multiplier > 0.0:
            e_losses = [
                self.einstein_losses[patch_idx].compute(
                    pts_limited[patch_idx],
                    metrics_limited[patch_idx],
                    model.patch_submodels[patch_idx],
                )
                for patch_idx in range(int(self.n_patches))
            ]
        else:
            e_losses = [
                tf.cast(0.0, tf.float64) for patch_idx in range(int(self.n_patches))
            ]
        if self.overlap_multiplier > 0.0 and self.n_patches > 1:
            overlap_loss = self.overlap_loss.compute(pts_overlap, metrics_overlap)
        else:
            overlap_loss = tf.cast(0.0, tf.float64)

        # Return loss components
        loss_constituents = {
            "einstein_losses": [tf.get_static_value(e_loss) for e_loss in e_losses],
            "overlap_loss": tf.get_static_value(overlap_loss),
        }

        # Compute the total loss (accounting for multiplier)
        global_loss = 0.0
        if self.einstein_multiplier > 0.0:
            global_loss += self.einstein_multiplier * tf.reduce_sum(
                tf.math.abs(e_losses)
            )
        if self.overlap_multiplier > 0.0:
            global_loss += self.overlap_multiplier * tf.math.abs(overlap_loss)
        global_loss /= self.einstein_multiplier + self.overlap_multiplier

        return global_loss, loss_constituents, sample_sizes
