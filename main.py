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

from dot_to_function import dot_to_function
from block_new import UpdatedBlock
from block_array import UpdatedBlock_arr
from pyadjoint.overloaded_function import overload_function
from numpy_block_var import ndarray
#from pyadjoint.overloaded_type import overload_function
#from pyadjoint.overloaded_function import overload_function
dot_to_function = overload_function(dot_to_function, UpdatedBlock)
#listinput = overload_function(list,UpdatedBlock_arr)

parser = argparse.ArgumentParser()
parser.add_argument("-f", "--forward", action="store_true", help="solve the forward problem")
parser.add_argument("-i", "--inverse", action="store_true", help="solve the inverse problem")
parser.add_argument("-o", "--observation", action="store_true", help="make the observation")
parser.add_argument("-r", "--regularization", type=int, default=0, help="power of regularization term ")
parser.add_argument("-nc", "--number_of_components", type=int, default=10, help="number of components in Truncated SVD")

if __name__ == "__main__":
	
	Discretization.add_args(parser)
	Observation.add_args(parser)
	args = parser.parse_args()
	
	disc = Discretization(args)
	#misfit = Misfit(args, disc)
	misfit = Misfit(args, disc, name="original")
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
	
	#Residual, Control = misfit.make_misfit(obs.observed)
	Residual = misfit.make_misfit(obs.observed, state.ka)
	Reg = Regularization(disc, state.ka)
	# Combine Misfit and Regularization, solve optimization problem
	Equation = Residual + Reg.reg

	Jhat = misfit.misfit(Equation, Control(state.ka))
    # misfit.misfit calls Control(ka)
	
	# problem = MinimizationProblem(Jhat, bounds=(0.0, 1.0))
	# parameters = {"acceptable_tol": 1.0e-3, "maximum_iterations": 100}
	# solver = IPOPTSolver(problem, parameters=parameters)
	# ka_opt = solver.solve()

	# xdmf_filename = XDMFFile("output/final_solution_Alpha(%f)_p(%f).xdmf" % (Reg.Alpha, Reg.power))
	# xdmf_filename.write(ka_opt)
	
	# At optimal point, we do partial SVD, get the vectors
	n_components = 50
	n_iter = 100
	# import pdb
	# pdb.set_trace()
	
	#U, Sigma, VT = randomized_svd1(Jhat, n_components= n_components, n_iter= n_iter, size = (disc.n+1)*(disc.n+1))
	#np.savetxt('U.txt', U)
	#np.savetxt('Sigma.txt', Sigma)
	#np.savetxt('VT.txt', VT)
	
	U = np.loadtxt('U.txt')
	Sigma = np.loadtxt('Sigma.txt')
	VT = np.loadtxt('VT.txt')

	# With vector, we can define the problem we're interested in:
	#import pdb
	#pdb.set_trace()
    # TODO: original prediction that operates on full parameter space (could even be an instance of Misfit)	
	prediction = Misfit(args, disc, name="prediction")
	# Residual1 = prediction.make_misfit(obs.observed,state.ka)
	
	#intermediate = U.T.dot(state.ka.vector()[:])
	intermediate = np.random.rand(n_components)
	ka_new = dot_to_function(state.A,U,intermediate)

	Residual2 = prediction.make_misfit(obs.observed,ka_new)
	#Reg1 = Regularization(disc, ka1)
	# Equation1 = Residual1
	Equation2 = Residual2
	# Jhat1 = prediction.misfit(Equation1, Control(state.ka
	ai = ndarray(intermediate.shape,buffer=intermediate)
	Jhat2 = prediction.misfit(Equation2, Control(ai))
	



	lb = 0.0
	ub = 1.0
	problem1 = MinimizationProblem(Jhat2, bounds=(lb, ub))

	parameters = {"acceptable_tol": 1.0e-3, "maximum_iterations": 100}
	import pdb
	pdb.set_trace()
	solver1 = IPOPTSolver(problem1, parameters=parameters)
	ka_opt1 = solver1.solve()

	xdmf_filename = XDMFFile("reduced_solution.xdmf")
	xdmf_filename.write(ka_opt1)

	import pdb
	pdb.set_trace()
# Projection
    # Interpolate

    # TODO: get a pyadjoint block for applying U (Pyadjoint block for numpy array)
    # U_pa (for PyAdjoint)
    # pred_value = prediction.misfit(U_py.apply(m_enc))
	
    # if we do this right, then we can get a ReducedFuntional for Control(m_enc)

    # TODO: define a encoded version of the prediction, that uses the full prediction on the expanded/decoded state
    # TODO: get reducedFunction block for encoded prediction and run optimization problem
	
