import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tensorflow as tf
tf.config.set_visible_devices([], "GPU")

import tensorflow as tf
tfk = tf.keras
tfk.backend.set_floatx("float64")

import numpy as np
import yaml
import wandb

from helper_functions import argument_parser, wandb_helper
from network.ball import BallNetwork
from sampling.ball import BallSample, CubeSample
from geometry.common import compute_scalar_curvature, compute_ricci_tensor  # optional


def main(hyperparameters_file, runtime_args, wandb_id=None):
    ###########################################################################
    # Load base hyperparameters from YAML
    with open(hyperparameters_file, "r") as file:
        args = yaml.safe_load(file)

    # Override with runtime args (CLI)
    for arg, arg_val in runtime_args.items():
        if arg in args:
            args[arg] = arg_val

    # Restore from WandB checkpoint?
    if wandb_id is not None:
        args, x_train = wandb_helper.restore_wandb(args, wandb_id)
    else:
        x_train, x_val = None, None

    ###########################################################################
    # Seed setup
    rng = np.random.default_rng()
    if args["np_seed"] is None:
        args["np_seed"] = int(rng.integers(2**32 - 2))
    if args["tf_seed"] is None:
        args["tf_seed"] = int(rng.integers(2**32 - 2))
    args["np_seed"] = int(args["np_seed"])
    args["tf_seed"] = int(args["tf_seed"])

    np.random.seed(args["np_seed"])
    tf.random.set_seed(args["tf_seed"])
    tfk.utils.set_random_seed(args["tf_seed"])

    print("TF random key: ", tf.random.uniform(shape=[6]))
    print("NP random key: ", np.random.randint(1, np.iinfo(int).max, size=6))

    ###########################################################################
    # WandB init
    wandb.init(project="Ainstein_ball", config=args, id=wandb_id, resume="allow")

    # Use wandb.config (NOT dict copy) so downstream code that expects attribute access still works.
    hp = wandb.config

    # Add run identifiers & conformal keys into wandb.config so all modules see them
    run_name = wandb.run.name or ""
    run_id = wandb.run.id or 42
    wandb.config.update(
        {
            "run_identifiers": (run_name, run_id),
            "conformal_mode": "L2",          
            "base_metric_kind": "round",
        },
        allow_val_change=True,
    )
    hp = wandb.config

    ###########################################################################
    # Data set-up
    if wandb_id is None:
        if hp["ball"]:
            train_sample = BallSample(
                hp["num_samples"],
                dimension=hp["dim"],
                patch_width=hp["patch_width"],
                density_power=hp["density_power"],
            )
            if hp["validate"]:
                val_sample = BallSample(
                    hp["num_val_samples"],
                    dimension=hp["dim"],
                    patch_width=hp["patch_width"],
                    density_power=hp["density_power"],
                )
        else:
            assert hp["n_patches"] == 1, (
                "Cube sampling only for local, single-patch geometries."
            )
            train_sample = CubeSample(
                hp["num_samples"],
                dimension=hp["dim"],
                width=hp["patch_width"],
                density_power=hp["density_power"],
            )
            if hp["validate"]:
                val_sample = CubeSample(
                    hp["num_val_samples"],
                    dimension=hp["dim"],
                    width=hp["patch_width"],
                    density_power=hp["density_power"],
                )
    else:
        train_sample = x_train
        if hp["validate"]:
            val_sample = x_val

    # Convert to tf tensors
    train_sample_tf = tf.convert_to_tensor(train_sample, dtype=tf.float64)
    val_sample_tf = None
    if hp["validate"]:
        val_sample_tf = tf.convert_to_tensor(val_sample, dtype=tf.float64)

    ###########################################################################
    # Instantiate and train network 
    network = BallNetwork(hp=hp, print_losses=hp["print_losses"])
    loss_hist = network.train(
        x_train=train_sample_tf, validate=hp["validate"], x_val=val_sample_tf
    )

    wandb.finish()
    return loss_hist, train_sample_tf, val_sample_tf


if __name__ == "__main__":
    args = argument_parser.get_args()
    wandb_id = args.wandb_id
    runtime_args = argument_parser.prune_none_args(args)
    lh, train_data, val_data = main(args.hyperparams, runtime_args, wandb_id)
