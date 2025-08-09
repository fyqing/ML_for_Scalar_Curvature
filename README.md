# ML package for the Nirenberg problem
This repository contains code for learning Riemannian metrics with prescribed scalar curvature function on sphere $S^2$. In particular, we fix the round metric, and search within its pointwise conformal class. This is known as the Nirenberg problem.
  
The package is run via the file `run.py`, where manifold properties and training hyperparameters are set using the `hyperparameters/hps.yaml` file. 

We recommend setting up a new environment for running of this package, the process for this is described in `environment/README.md`.  

## Running from the command line  
To run from the command line, enter the local directory of this package, ensure the environment is activated, set the run hyperparameters in `hyperparameters/hps.yaml`, and run the following code:  
### If using Weights & Biases:
```
python3 run.py --hyperparams=hyperparameters/hps.yaml
```
### ...otherwise:
```
wandb disabled
python3 run.py --hyperparams=hyperparameters/hps.yaml
```

### Functionality
The package functionality is split according to: the model in `network/`, the losses in `losses/`, the sampling in `sampling/`, the geometric functions in `geometry/`, and some additional useful functions in `helper_functions/helper_functions.py`. The models are saved into the `runs/` folder (the local filepath to this must first be set in `hps.yaml`).

A jupyter notebook `examine_output.ipynb` is provided which provides the testing functionality, and allows interactive visualisation of the trained models. Ensure the local filepath to the trained models is set correctly and follow internal instructions to set up the testing.   
  

