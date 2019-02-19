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


from block_new import UpdatedBlock
from block_array import UpdatedBlock_arr
from pyadjoint.overloaded_function import overload_function
from numpy_block_var import ndarray

## We already defined our customized function dot_to_function.
## Using this function, we could successfuly use adjoint equation to estimate the parameters. 
from dot_to_function import dot_to_function
dot_to_function = overload_function(dot_to_function, UpdatedBlock)

## We can change the setting on the command line.
parser = argparse.ArgumentParser()
parser.add_argument("-f", "--forward", action="store_true", help="solve the forward problem")
parser.add_argument("-i", "--inverse", action="store_true", help="solve the inverse problem")
parser.add_argument("-o", "--observation", action="store_true", help="make the observation")
parser.add_argument("-r", "--regularization", type=int, default=0, help="power of regularization term ")
parser.add_argument("-nc", "--number_of_components", type=int, default=10, help="number of components in Truncated SVD")

if __name__ == "__main__":
    
    tape = get_working_tape()
   
    ## We can change the settings in direcretization and   
    Discretization.add_args(parser)   
    Observation.add_args(parser)
  
    ## Getting input values
    args = parser.parse_args()
    
    ## First of solving the problems, we defined the discretization of the problem.    
    disc = Discretization(args)
    
    ## Next we defined the Misifit using discretization.
    misfit = Misfit(args, disc, name="original")
    

    ## Next we got the state variables and observation from the misfit.
    state = misfit.state
    obs = misfit.obs

    ## If we needed to make the observation, we could make the observation
    ## Otherwise we could just load the data from the files.
    if args.observation:
        K = get_coefficient_space(disc.mesh)
        ka_true = get_initial_coefficients(K)
        w_true = state.solve(ka=ka_true)
        obs.set_observed(w_true)
    else:
        obs.get_observed(state.W)
     
    ## Next we dfined residual and regularization
    residual = misfit.make_misfit(obs.observed, state.ka)
    reg = Regularization(disc, state.ka)
    
    ## Next we combined misfit and regularization to define reduced functional objective
    objective = residual + reg.reg
    with tape.name_scope("misfit_first_part"):
        Jhat = misfit.misfit_op(objective, Control(state.ka))
    
    ######################################################
    ## Sovling minimization problem and save the result ##
    with tape.name_scope("minimization_first_part"):
        problem = MinimizationProblem(Jhat, bounds=(0.0, 1.0))
        parameters = {"acceptable_tol": 1.0e-3, "maximum_iterations": 5}
        solver = IPOPTSolver(problem, parameters=parameters)
        ka_opt = solver.solve()

    # xdmf_filename = XDMFFile("output/final_solution_Alpha(%f)_p(%f).xdmf" % (Reg.Alpha, Reg.power))
    # xdmf_filename.write(ka_opt)
    ######################################################

    ## At optimal point, we do partial SVD, get the vectors
    n_components = 50
    n_iter = 100   
    # U, Sigma, VT = randomized_svd1(Jhat, n_components= n_components, n_iter= n_iter, size = (disc.n+1)*(disc.n+1))
    
    # np.savetxt('U.txt', U)
    # np.savetxt('Sigma.txt', Sigma)
    # np.savetxt('VT.txt', VT)
    
    U = np.loadtxt('U.txt')
    Sigma = np.loadtxt('Sigma.txt')
    VT = np.loadtxt('VT.txt')

    # With vector, we can define the problem we're interested in:
    # TODO: original prediction that operates on full parameter space (could even be an instance of Misfit)    
    with tape.name_scope("Define_prediction"):
        prediction = Misfit(args, disc, name="prediction")
        # Residual1 = prediction.make_misfit(obs.observed,state.ka)
    
    with tape.name_scope("Make_first_guess"):    
        intermediate = np.random.rand(n_components)

    with tape.name_scope("Making_taped_array"):
        ai = ndarray(intermediate.shape,buffer=intermediate)
    
    with tape.name_scope("putting_into_defined_function"):
        ka_new = dot_to_function(state.A,U,ai)

    with tape.name_scope("Making_residual"):
        residual2 = prediction.make_misfit(obs.observed,ka_new)
    #Reg1 = Regularization(disc, ka1)
    # Equation1 = Residual1
    with tape.name_scope("Defining_objective"):
        objective2 = residual2
    # Jhat1 = prediction.misfit(Equation1, Control(state.ka))
    with tape.name_scope("Make_reduced_functional"):
        Jhat2 = prediction.misfit_op(objective2, Control(ai))

    
    lb = 0.0
    ub = 1.0
    with tape.name_scope("Minimization_problem_setting"):
        problem1 = MinimizationProblem(Jhat2, bounds=(lb, ub))
        parameters = {"acceptable_tol": 1.0e-2, "maximum_iterations": 5}
    # import pdb
    # pdb.set_trace()
    with tape.name_scope("Defining_minimization"):
        solver1 = IPOPTSolver(problem1, parameters=parameters)
    
    with tape.name_scope("Solving"):
        ka_opt1 = solver1.solve()

    tape.visualise()
    
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
    
