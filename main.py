import argparse
import os 

from fenics import *
from fenics_adjoint import *
import sympy as sym
import moola
import numpy as np
import math

from scipy import linalg, sparse
from sklearn.utils import *
from sklearn.utils.extmath import svd_flip
from SVD import safe_sparse_dot, randomized_svd, randomized_range_finder
from initialize import *

from Inverse import *
from discretization import Discretization
from state import State
from misfit import Misfit
from observation import Observation
from regularization import Regularization

parser = argparse.ArgumentParser()
parser.add_argument("-f", "--forward", action="store_true", help="solve the forward problem")
parser.add_argument("-i", "--inverse", action="store_true", help="solve the inverse problem")
parser.add_argument("-o", "--observation", action="store_true", help="make the observation")
parser.add_argument("-a", "--add_noise", type=int, default=0, help="add noise with specific standard deviation")
parser.add_argument("-r", "--regularization", type=int, default=0, help="power of regularization term ")


if __name__ == "__main__":
	
	Discretization.add_args(parser)
	# Misfit.add_args(parser)
	# Regularization.add_args(parser)
	args = parser.parse_args()
	disc = Discretization(args)

	misfit = Misfit(args, disc)
	state = misfit.state
	obs = misfit.obs

	import pdb
	pdb.set_trace()
	# create the Misfit object from args, disc Misfit(args,disc)
	# From the misfit object, get the state object (misfit.state, describing state equtions) and observation object (misfit.obs, describing observation process)
	
	if args.observation:
		# Use the state object to solve for state
		# Use the observation object to save noisy obervations
		observation = Observation(disc)
		# obs.set_data(noise)

	# state = State(disc) # TODO: state is a component of misfit (misfit.state)
	# reg = Regularization(state) # TODO: Regularization(args,disc,state.ka)
	
	import pdb
	pdb.set_trace()

	# Combine Misfit and Regularization, solve optimization problem

	# At optimal point, we do partial SVD, get the vectors

	# With vector, we can define the problem we're interested in:
	# Define a new optimization problem using prediction 
		