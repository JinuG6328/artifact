import argparse
import os 

from fenics import *
from fenics_adjoint import *
import sympy as sym
import numpy as np
import math
import matplotlib.pyplot as plt

from scipy import linalg, sparse
from sklearn.utils import *
from sklearn.utils.extmath import svd_flip
from pyadjoint.overloaded_function import overload_function

from SVD_extra import get_matrix, safe_sparse_dot, randomized_range_finder, randomized_svd1, reject_outlier
from covariance import PriorPrecHessian
from initialize import *
from discretization import Discretization
from state import State
from misfit import Misfit
from observation import Observation
from regularization import Regularization
from block_new import UpdatedBlock

from numpy_block_var import Ndarray

## We already defined our customized function dot_to_function.
## Using this function, we could successfuly use adjoint equation to estimate the parameters. 
from dot_to_function import dot_to_function
dot_to_function = overload_function(dot_to_function, UpdatedBlock)

## We can change the setting on the command line
parser = argparse.ArgumentParser()
parser.add_argument("-f", "--forward", action="store_true", help="solve the forward problem")
parser.add_argument("-i", "--inverse", action="store_true", help="solve the inverse problem")
parser.add_argument("-o", "--observation", action="store_true", help="make the observation")
parser.add_argument("-r", "--regularization", type=int, default=0, help="power of regularization term ")
parser.add_argument("-nc", "--number_of_components", type=int, default=10, help="number of components in Truncated SVD")

if __name__ == "__main__":

    # set_log_level(16)
    
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
    residual_red = misfit.make_misfit_red(obs.observed, state.ka)
    residual = misfit.make_misfit(obs.observed, state.ka)
    reg = Regularization(state.ka, state.A)
    
    ## Next we combined misfit and regularization to define reduced functional objective
    objective = residual_red + reg.reg
    Jhat = misfit.misfit_op(objective, Control(state.ka))

    ## Sovling minimization problem and save the result
    problem = MinimizationProblem(Jhat, bounds=(0.0, 1.0))
    parameters = {"acceptable_tol": 1.0e-4, "maximum_iterations": 50}
    solver = IPOPTSolver(problem, parameters=parameters)
    ka_opt = solver.solve()
    # xdmf_filename = XDMFFile("output/final_solution_Alpha(%f)_p(%f).xdmf" % (Reg.Alpha, Reg.power))
    # xdmf_filename.write(ka_opt)

    import pdb
    pdb.set_trace()


    ## Taylor test
    # conv_rate = taylor_test(sol_residual, state.ka, state.ka*0.1)

    ## Making Jhat_red from misfit only 
    Jhat_red = misfit.misfit_op(residual_red, Control(state.ka))

    ## Calculating PriorPreconditionedHessian matrix of Jhat_red
    priorprehessian = PriorPrecHessian(Jhat_red, reg, state.ka)    

    ## Number of components, number of iteration, and randomized SVD
    n_components = 20#100
    n_iter = 20#00   
    U, Sigma, VT = randomized_svd1(priorprehessian, n_components= n_components, n_iter= n_iter, size = (disc.n+1)*(disc.n+1))

    ##########################################################
    ## With U(VT), we can define the reduced space problem: ##
    ##########################################################
    
    ## Making prediction object to use observations
    prediction = Misfit(args, disc, name="prediction")

    ## Initializing the array to optimize
    intermediate = np.random.rand(n_components)
    ai = Ndarray(intermediate.shape, buffer=intermediate)
        
    ## Converting the array into function
    ka_new = dot_to_function(state.A, U, ai)
    ka_new.vector()[:] += ka_opt.vector()[:]
    # ka_new = dot_to_function(state.A, VT.T, ai)
    # ka_new_norm = assemble(dot(ka_new,ka_new)*dx)

    ## Making_residual with full space
    objective2 = prediction.make_misfit(obs.observed, ka_new)

    ## Making Jhat2
    Jhat2 = prediction.misfit_op(objective2, Control(ai))

    ## Solve the optimization problem
    ## TODO constraints
    # constraints = UFLInequalityConstraint((V/delta - rho), ai)
    problem1 = MinimizationProblem(Jhat2)
    parameters = {"acceptable_tol": 1.0e-4, "maximum_iterations": 50}
    solver1 = IPOPTSolver(problem1, parameters=parameters)
    ka_opt1 = solver1.solve()     
    

    ## Save the result using existing program tool.
    ka_opt2 = ka_opt.copy(deepcopy = True)
    
    # ka_opt2.vector()[:] = reject_outlier(U.dot(ka_opt1))
    ka_opt2.vector()[:] = U.dot(ka_opt1)
    print("Norm %f", np.linalg.norm(U.dot(ka_opt1)))

    firstplot = plot(ka_opt2)
    plt.colorbar(firstplot, ticks = [-0.1, 0, 1, 10, 50])
    plt.figure()
    secondplot = plot(ka_opt)
    plt.colorbar(secondplot, ticks = [0, 0.49, 0.5, 0.75, 1])  
    plt.figure()
    plt.plot(Sigma)
    plt.show()
    import pdb
    pdb.set_trace()
    #np.savetxt('Sigma.txt', Sigma)

    # xdmf_filename = XDMFFile("reduced_solution.xdmf")
    # xdmf_filename.write(ka_opt2)
