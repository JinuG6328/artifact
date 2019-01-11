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
	args = parser.parse_args()
	disc = Discretization(args)
	
	if args.observation:
		observation = Observation(disc)

	state = State(disc)
	reg = Regularization(state)
	misfit = Misfit(state)
	import pdb
	pdb.set_trace()
		