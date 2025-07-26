import tensorflow as tf
import numpy as np

tf.keras.backend.set_floatx("float64")


###############################################################################
# Functions to change between patches

def patch_xy_to_xyz(coords, patch_idx):
    """
    Stereographic inverse: patch coords -> point on S^2 in R^3.
    patch_idx = 0 (North), 1 (South).
    """
    x1, x2 = coords[:, 0], coords[:, 1]          
    r2 = x1**2 + x2**2

    denom = 1.0 + r2
    X = 2.0 * x1 / denom
    Y = 2.0 * x2 / denom
    Z = (1.0 - r2) / denom
    if patch_idx == 1:  # South chart flips
        Z = -Z
    return tf.stack([X, Y, Z], axis=-1)

# Prescribed K functions

def prescribed_K(coords, kind="zero", patch_idx=0):
    """
    Returns a prescribed scalar curvature K at each point, based on the kind.

    Args:
        coords (tf.Tensor): Coordinates of shape [batch_size, dim].
        kind (str): Type of K function to use.

    Returns:
        tf.Tensor: K values at each point (shape [batch_size]).
    """
    xyz = patch_xy_to_xyz(coords, patch_idx)    
    x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]       

    theta  = tf.acos(tf.clip_by_value(z, -1.0, 1.0)) 
    phi = tf.math.atan2(y, x)
    phi = tf.where(phi < 0.0, phi + 2.0 * np.pi, phi)

    if kind == "zero":
        return tf.zeros_like(theta)
    elif kind == "cos2_theta_plus_offset":
        return tf.math.cos(theta)**2 + 0.2
    elif kind == "gaussian_bump":
        return tf.exp(-20 * (theta - np.pi / 2) ** 2) + 0.1
    elif kind == "gaussian":
        A = 1.0                                 
        const = 4*np.pi - A*np.pi*np.sqrt(np.pi/20)  
        return A*tf.exp(-20*(theta - np.pi/2)**2) + const/(4*np.pi)
    elif kind == "double_peak":
        return tf.exp(-30 * (theta - 1.0) ** 2) + tf.exp(-30 * (theta - 2.0) ** 2)
    elif kind == "round":
        return tf.constant(2.0, dtype=tf.float64) * tf.ones_like(theta)
    elif kind == "step_equator":
        return 1.0 + 0.5 * tf.tanh(10.0 * (np.pi / 2 - theta))
    elif kind == "sinusoidal":
        return 1.0 + 0.5 * tf.sin(3.0 * theta)
    elif kind == "north_pole_bump":
        return tf.exp(-40.0 * theta**2)
    elif kind == "tilted_step_smooth":
        return 0.3 + 0.9 * tf.sigmoid(20.0 * (z - 0.3))
    elif kind == 'cos':
        return tf.math.cos(theta)
    elif kind == '3cos2':
        return 3 * tf.math.cos(theta) ** 2 - 1
    elif kind == 'sin_cos':
        return tf.math.sin(theta) ** 2 * tf.math.cos(2 * phi)
    elif kind == '5cos3':
        return 5 * tf.math.cos(theta) ** 3 - 3 * tf.math.cos(theta)
    elif kind == '5cos3_1':
        return 5 * tf.math.cos(theta) ** 3 - 3 * tf.math.cos(theta) + 1
    elif kind == 'sincoscos':
        return tf.math.sin(theta) * tf.math.cos(theta) * tf.math.cos(phi)
    else:
        raise ValueError(f"Unknown scalar_target_type: {kind}")


def build_conformal_metric_global(global_u_fn, coords, patch_idx=0, kind="round"):
    """
    coords: [batch,2] patch coords
    global_u_fn: callable xyz->[batch]  (u evaluated on S^2)
    return: [batch,2,2] metric in patch coords
    """
    xyz = patch_xy_to_xyz(coords, patch_idx)
    u_vals = global_u_fn(xyz)         # [batch]
    g0 = AnalyticMetric_Ball(coords)
    e2u = tf.exp(2.0 * u_vals)[:, None, None]
    return e2u * g0

# Ball coordinates
def PatchChange_Coordinates_Ball(coords):
    # Compute the coordinate norm
    norm = tf.norm(coords, axis=1)

    # Compute the patch transformation
    coords_otherpatch = coords * tf.expand_dims(
        (norm - 1) / (norm * (norm + 1)), axis=-1
    )

    return coords_otherpatch


def PatchChange_Metric_Ball(coords, metric_pred):
    # Change the coordinates to the other patch
    coords_otherpatch = PatchChange_Coordinates_Ball(coords)

    # Compute the coordinate norm
    norm = tf.norm(coords_otherpatch, axis=1)

    # Compute the Jacobian
    jacobian_term1 = tf.eye(
        coords_otherpatch.shape[1],
        batch_shape=[coords_otherpatch.shape[0]],
        dtype=coords_otherpatch.dtype,
    )
    jacobian_term1 *= tf.expand_dims(
        tf.expand_dims((norm - 1) / (norm * (norm + 1)), axis=-1), axis=-1
    )
    jacobian_term2 = tf.einsum("si,sj->sij", coords_otherpatch, coords_otherpatch)
    jacobian_term2 *= tf.expand_dims(
        tf.expand_dims((1 + 2 * norm - tf.square(norm)) / (tf.pow(norm, 3) * tf.square(1 + norm)), axis=-1),
        axis=-1,
    )
    jacobian = jacobian_term1 + jacobian_term2

    # Compute the patch transformation
    metric_otherpatch = tf.einsum("sij,sjk,skl->sil", jacobian, metric_pred, jacobian)

    return metric_otherpatch


# Stereographic coordinates
def PatchChange_Coordinates_Stereo(coords):
    # Compute the coordinate norm
    norm = tf.norm(coords, axis=1)

    # Compute the patch transformation
    coords_otherpatch = coords / tf.expand_dims(tf.square(norm), axis=-1)

    return coords_otherpatch


def PatchChange_Metric_Stereo(coords, metric_pred):
    # Change the coordinates to the other patch
    coords_otherpatch = PatchChange_Coordinates_Stereo(coords)

    # Compute the coordinate norm
    norm = tf.norm(coords_otherpatch, axis=1)

    # Compute the Jacobian
    jacobian_term1 = tf.eye(
        coords_otherpatch.shape[1],
        batch_shape=[coords_otherpatch.shape[0]],
        dtype=coords_otherpatch.dtype,
    )
    jacobian_term1 /= tf.expand_dims(tf.expand_dims(tf.square(norm), axis=-1), axis=-1)
    jacobian_term2 = tf.einsum("si,sj->sij", coords_otherpatch, coords_otherpatch)
    jacobian_term2 *= tf.expand_dims(tf.expand_dims(-2 / tf.pow(norm, 4), axis=-1), axis=-1)
    jacobian = jacobian_term1 + jacobian_term2

    # Compute the patch transformation
    metric_otherpatch = tf.einsum("sij,sjk,skl->sil", jacobian, metric_pred, jacobian)

    return metric_otherpatch


# Define function to compute the analytic round metric at input ball points
def AnalyticMetric_Ball(coords, identity=False):
    # Return the identity function if requested
    if identity:
        return tf.eye(
            coords.shape[1], batch_shape=[coords.shape[0]], dtype=coords.dtype
        )

    # Otherwise compute the round metric
    norm = tf.norm(coords, axis=1)

    metric_term1 = tf.eye(
        coords.shape[1], batch_shape=[coords.shape[0]], dtype=coords.dtype
    )
    metric_term1 *= tf.expand_dims(
        tf.expand_dims(16 * tf.square(1 - tf.square(norm)), axis=-1), axis=-1
    )
    metric_term2 = 64 * tf.einsum("si,sj->sij", coords, coords)
    metric = metric_term1 + metric_term2
    metric /= tf.expand_dims(tf.expand_dims(tf.pow(1 + tf.square(norm), 4), axis=-1), axis=-1)

    return metric
