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
from ipopt_solver1 import *
from resi_const import *

from covariance import PriorPrecHessian
from initialize import *
from discretization import Discretization
from state import State
from misfit import *
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
parser.add_argument("-o", "--observation", action="store_true", help="make the observation")
parser.add_argument("-r", "--regularization", type=int, default=0, help="power of regularization term ")
parser.add_argument("-nc", "--number_of_components", type=int, default=20, help="number of components in Truncated SVD")
parser.add_argument("-ni", "--number_of_iterations", type=int, default=20, help="number of power iterations in Truncated SVD")
parser.add_argument("-nm", "--number_of_iterations_for_model", type=int, default=50, help="number of iterations for optimization")
parser.add_argument("-nb", "--number_of_iterations_for_boundary", type=int, default=30, help="number of iterations for getting pressure boundary")
parser.add_argument("-rb", "--reduced_boundary", action="store_true", help="pressure boundary from the reduced space otherwise that from full space")

if __name__ == "__main__":
 
    ## We can change the settings in direcretization and   
    Discretization.add_args(parser)   
    Observation.add_args(parser)
    Regularization.add_args(parser)
    Misfit.add_args(parser)
  
    ## Getting input values
    args = parser.parse_args()
    
    ## First of solving the problems, we defined the discretization of the problem.    
    disc = Discretization(args)
    
    ## Next we defined the Misifit using discretization.
    # Want a flag to say which original inverse problem to run, save that information in misfit
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
     
    #########################################################################
    ## Inverse problem with full space and 9 componets observation ##########
    #########################################################################

    ## Next we dfined regularization
    reg = Regularization(state.ka, state.A, args)

    ## Next we combined misfit and regularization to define reduced functional objective
    objective = misfit(state.ka, obs.observed) + reg(state.ka)
    
    Jhat = ReducedFunctional_(objective, Control(state.ka))

    # Sovling minimization problem and save the result
    problem = MinimizationProblem(Jhat)
    parameters = {"acceptable_tol": 1.0e-3, "maximum_iterations": 50}
    solver = IPOPTSolver(problem, parameters=parameters)   
    ka_opt = solver.solve()
    
    # xdmf_filename = XDMFFile("output/final_solution_Alpha(%f)_p(%f).xdmf" % (Reg.Alpha, Reg.power))
    # xdmf_filename.write(ka_opt)

    ## Taylor test
    # conv_rate = taylor_test(sol_residual, state.ka, state.ka*0.1)

    #########################################################################
    ## Prior Preconditioned Hessian Randomized SVD ##########################
    #########################################################################

    ## Calculating PriorPreconditionedHessian matrix of Jhat_red
    priorprehessian = PriorPrecHessian(Jhat, reg, state.ka)    

    ## Number of components, number of iteration, and randomized SVD
    n_components = args.number_of_components
    n_iter = args.number_of_iterations
    U, Sigma, VT = randomized_svd1(priorprehessian, n_components= n_components, n_iter= n_iter, size = (disc.n+1)*(disc.n+1))

    #########################################################################
    ## With U(VT), we can define the reduced space problem: #################
    ## Inverse problem with reduced space and full observation ##############
    #########################################################################

    ## Making prediction object to use observations
    misfit_red = Misfit(args, disc, name="misfit_reduced_space")

    ## Initializing the array to optimize
    intermediate = np.random.rand(n_components)
    ai = Ndarray(intermediate.shape, buffer=intermediate)
        
    ## Converting the array into function
    ka_new = dot_to_function(state.A, U, ai)
    ka_new_opt = ka_new + ka_opt
    
    ## Next we combined misfit and regularization to define reduced functional objective
    objective2 = misfit_red(ka_new_opt, obs.observed)
    
    ## Making Jhat2
    Jhat2 = ReducedFunctional_(objective2, Control(ai))

    iters = args.number_of_iterations_for_model

    ## Solve the optimization problem
    problem1 = MinimizationProblem(Jhat2, constraints=ResidualConstraint(1, Jhat, U))
    parameters = {"acceptable_tol": 1.0e-3, "maximum_iterations": iters}
    solver1 = IPOPTSolver(problem1, parameters=parameters)
    ka_opt1 = solver1.solve()     
    
    switch = args.reduced_boundary
    if switch == False:
    #########################################################################
    ## Finding the range of the pressure at specific point with full space###
    #########################################################################
        n_iters = args.number_of_iterations_for_boundary

        ## Pressure at the 0.5, 0.8    
        prediction = Misfit(args, disc, name="prediction")
        pressure_cen = prediction(state.ka)
        Jhat_cen = ReducedFunctional_(pressure_cen, Control(state.ka))

        problem_pred_low = MinimizationProblem(Jhat_cen, constraints=ResidualConstraint(1, Jhat))
        parameters = {"acceptable_tol": 1.0e-3, "maximum_iterations": n_iters_f}
        solver_pred_low = IPOPTSolver(problem_pred_low, parameters=parameters)
        ka_pred_low = solver_pred_low.solve()

        problem_pred_up = MaximizationProblem(Jhat_cen, constraints=ResidualConstraint(1, Jhat))
        parameters = {"acceptable_tol": 1.0e-3, "maximum_iterations": n_iters_f}
        solver_pred_up = IPOPTSolver(problem_pred_up, parameters=parameters)
        ka_pred_up = solver_pred_up.solve()

        prediction_upper_bound = Jhat_cen(ka_pred_up)
        prediction_lower_bound = Jhat_cen(ka_pred_low)


    #########################################################################
    ## Finding the range of the pressure at specific point ##################
    ## With reduced space ###################################################
    #########################################################################
    else:
        n_iters = args.number_of_iterations_for_boundary

        intermediate = np.random.rand(n_components)
        random_array = Ndarray(intermediate.shape, buffer=intermediate)
            
        ## Converting the array into function
        ka_new_red = dot_to_function(state.A, U, random_array)
        ka_new_opt_red = ka_new_red + ka_opt
        pressure_cen = prediction(ka_new_opt_red)
        Jhat_cen_red = ReducedFunctional_(pressure_cen, Control(random_array))

        problem_pred_low_red = MinimizationProblem(Jhat_cen_red, constraints=ResidualConstraint(1, Jhat2, U))
        parameters = {"acceptable_tol": 1.0e-3, "maximum_iterations": 1}
        solver_pred_low_red = IPOPTSolver(problem_pred_low_red, parameters=parameters)
        ka_pred_low_red = solver_pred_low_red.solve()

        problem_pred_up_red = MaximizationProblem(Jhat_cen_red, constraints=ResidualConstraint(1, Jhat2, U))
        parameters = {"acceptable_tol": 1.0e-3, "maximum_iterations": 1}
        solver_pred_up_red = IPOPTSolver(problem_pred_up_red, parameters=parameters)
        ka_pred_up_red = solver_pred_up_red.solve()

        prediction_upper_bound = Jhat_cen_red(ka_pred_up_red)
        prediction_lower_bound = Jhat_cen_red(ka_pred_low_red)

    print(prediction_upper_bound)
    print(prediction_lower_bound)
    # print(prediction_upper_bound_red)
    # print(prediction_lower_bound_red)

    ## Save the result using existing program tool.
    ka_opt2 = ka_opt.copy(deepcopy = True)
    
    # ka_opt2.vector()[:] = reject_outlier(U.dot(ka_opt1))
    #ka_opt2.vector()[:] = U.dot(ka_opt1)
    ka_opt2.vector()[:] = U.dot(ka_opt1)
    # print("Norm %f", np.linalg.norm(U.dot(ka_opt1)))

    ## prediction
    pre_obs = prediction.obs
    pre_state = prediction.state
    pre_w = pre_state.solve(ka=ka_opt2)
    # import pdb
    # pdb.set_trace()
    pre_u, pre_p = pre_w.split(deepcopy=True)

    firstplot = plot(ka_opt2)
    plt.colorbar(firstplot, ticks = [-0.5, 0, 0.1, 0.25, 0.5, 1])
    plt.figure()
    secondplot = plot(ka_opt)
    plt.colorbar(secondplot, ticks = [-0.5, 0, 0.1, 0.25, 0.5, 1])  
    plt.figure()
    plt.plot(Sigma)
    plt.show()
    import pdb
    pdb.set_trace()
    #np.savetxt('Sigma.txt', Sigma)

    # xdmf_filename = XDMFFile("reduced_solution.xdmf")
    # xdmf_filename.write(ka_opt2)
