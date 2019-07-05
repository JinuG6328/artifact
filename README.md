# Fusing Optimal Uncertainty Quantification with Low-lank decomposition
  
## Description

This example calculate the pressure envelope at the specific point in full space and in reduced space, which comes from SVD. 
To be specific, we estimate the model parameter(permeability) from the observation(pressure), which is MAP point. 
We can also get a covariance of model parameter from the misfit. 
By decomposing the covariance matrix with SVD, we could get the subspaces, which the model parameters are in.
By using the MAP point and the subspaces, we can get the envelope of the pressure at specific point from full space and reduedced space, and compare it with each other. 


## Installing
We use Anaconda to make the environment.
By using the jgoenv.txt file, you can install the environment.

```
conda create --name jgoenv --file jgoenv.txt
```

## Running the test

```
conda activate jgoenv
python main.py -n 16 -so opt_sol -ss subspace
```
