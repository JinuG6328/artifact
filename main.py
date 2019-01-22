import argparse
import os 

from fenics import *
from fenics_adjoint import *
import sympy as sym
import moola
import numpy as np
import math
import matplotlib.pyplot as plt

from scipy import linalg, sparse
from sklearn.utils import *
from sklearn.utils.extmath import svd_flip
from SVD_extra import get_matrix, safe_sparse_dot, randomized_range_finder, randomized_svd1
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
parser.add_argument("-r", "--regularization", type=int, default=0, help="power of regularization term ")

if __name__ == "__main__":
	
	Discretization.add_args(parser)
	Observation.add_args(parser)
	args = parser.parse_args()
	
	disc = Discretization(args)
	misfit = Misfit(args, disc)
	state = misfit.state
	obs = misfit.obs

	# RHS requires v in state. 
	# If RHS is required, I could make it as a function inside the state.

	if args.observation:
        # Solve the state for some known RHS
		K = get_coefficient_space(disc.mesh)
		ka_true = get_initial_coefficients(K)
		w_true = state.solve(ka=ka_true)
		obs.set_observed(w_true)
	
	else:
		obs.get_observed(state.W)
	
	# import pdb
	# pdb.set_trace()
	
	Residual, Control = misfit.make_misfit(obs.observed)
	Reg = Regularization(disc, misfit.ka)
	Equation = Residual + Reg.reg
	Jhat = misfit.misfit(Equation, Control)
	
	# problem = MinimizationProblem(Jhat, bounds=(0.0, 1.0))
	# parameters = {"acceptable_tol": 1.0e-3, "maximum_iterations": 100}
	# solver = IPOPTSolver(problem, parameters=parameters)
	# ka_opt = solver.solve()

	# xdmf_filename = XDMFFile("output/final_solution_Alpha(%f)_p(%f).xdmf" % (Reg.Alpha, Reg.power))
	# xdmf_filename.write(ka_opt)
	
	n_components = 30
	n_iter = 10
	# import pdb
	# pdb.set_trace()
	U, Sigma, VT = randomized_svd1(Jhat, n_components= n_components, n_iter= n_iter, size = (disc.n+1)*(disc.n+1))
	
	np.savetext('U.txt', U)
	np.savetext('Sigma.txt', Sigma)
	np.savetext('VT.txt', VT)
	
	# import pdb
	# pdb.set_trace()


	# state = State(disc) # TODO: state is a component of misfit (misfit.state)
	# reg = Regularization(state) # TODO: Regularization(args,disc,state.ka)
	
	# import pdb
	# pdb.set_trace()

	# Combine Misfit and Regularization, solve optimization problem

	# At optimal point, we do partial SVD, get the vectors

	# With vector, we can define the problem we're interested in:
	# Define a new optimization problem using prediction 
		
