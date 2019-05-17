import argparse
import os 

from fenics import *
from fenics_adjoint import *
import numpy as np
import matplotlib.pyplot as plt

from scipy import linalg
from sklearn.utils import *
from sklearn.utils.extmath import svd_flip
from pyadjoint.overloaded_function import overload_function
from pyadjoint.tape import stop_annotating, get_working_tape

from SVD_extra import get_matrix, safe_sparse_dot, randomized_range_finder, randomized_svd1, reject_outlier
from ipopt_solver1 import *
from resi_const import *

from covariance import PriorPrecHessian
from initialize import *
from discretization import Discretization
from state import State
from misfit import Misfit, ReducedFunctional_
from prediction import Prediction
from observation import Observation
from regularization import Regularization
from block_new import UpdatedBlock

from numpy_block_var import Ndarray

## We already defined our customized function dot_to_function.
## Using this function, we could successfuly use adjoint equation to estimate the parameters. 
from dot_to_function import dot_to_function
dot_to_function = overload_function(dot_to_function, UpdatedBlock)

if __name__ == "__main__":

    ## We can change the setting on the command line
    parser = argparse.ArgumentParser()
    parser.add_argument("-nc", "--number_of_components", type=int, default=20, help="number of components in Truncated SVD")
    parser.add_argument("-ni", "--number_of_iterations", type=int, default=20, help="number of power iterations in Truncated SVD")
    parser.add_argument("-ne", "--number_of_extra_vectors", type=int, default=20, help="number of extra vectors in Truncated SVD")
    parser.add_argument("-i", "--check_inverse_problem", action="store_true", help="check the inverse problem result with 9 components")
    parser.add_argument("-p", "--prediction", action="store_true", help="check the pressure boundary of specific point")
    parser.add_argument("-nm", "--number_of_iterations_for_model", type=int, default=30, help="number of iterations for optimization")
    parser.add_argument("-nb", "--number_of_iterations_for_boundary", type=int, default=30, help="number of iterations for getting pressure boundary")
    parser.add_argument("-rb", "--reduced_boundary", action="store_true", help="pressure boundary from the reduced space otherwise that from full space")
    # TODO: read / write optimal ka
    # TODO: read / write subspaces

    ## We can change the settings in direcretization and   
    Discretization.add_args(parser)   
    Observation.add_args(parser)
    Regularization.add_args(parser)
    Misfit.add_args(parser)
  
    ## Getting input values
    args = parser.parse_args()
    
    ## Before solving the problems, we defined the discretization of the problem.    
    disc = Discretization(args)

    ## Context that describes what data we have and how it relates to the model
    obs = Observation(args, disc)

    ## Context that describes how observable state is determined by the parameters
    state = State(args, disc)

    ## Next we defined the Misfit using discretization.
    misfit = Misfit(args, disc, obs=obs, state=state)
    
    #########################################################################
    ## Inverse problem with full space and 9 componets observation ##########
    #########################################################################

    ## Next we define regularization
    reg = Regularization(args, disc)

    ## get a parameter vector
    with get_working_tape().name_scope("parameters"):
        ka = state.default_parameters()

    ## Next we combine misfit and regularization to define reduced functional objective
    m = misfit(ka)
    r = reg(ka)

    with get_working_tape().name_scope("objective"):
        objective = m + r

    Jhat = ReducedFunctional_(objective, Control(ka))

    # Solving minimization problem and save the result
    # TODO: options to skip optimization loop by reading from file
    with stop_annotating():
        problem = MinimizationProblem(Jhat)
        parameters = {"acceptable_tol": 1.0e-6, "maximum_iterations": 100}
        solver = IPOPTSolver(problem, parameters=parameters)   
        opt_sol = solver.solve().copy(deepcopy=True)

    with get_working_tape().name_scope("optimal_parameters"):
        ka_opt = opt_sol.copy(deepcopy=True)

    # TODO: options to write optimal parameters to file

    # TODO: options to read subspaces from file
    with stop_annotating():
        # xdmf_filename = XDMFFile("output/final_solution_Alpha(%f)_p(%f).xdmf" % (Reg.Alpha, Reg.power))
        # xdmf_filename.write(ka_opt)

        ## Taylor test
        # conv_rate = taylor_test(sol_residual, state.ka, state.ka*0.1)

        #########################################################################
        ## 9 components observation and Randomized SVD ##########################
        #########################################################################

        Jhat_m = misfit.misfit_op(m, Control(ka))

        ## Calculating PriorPreconditionedHessian matrix of Jhat_m
        priorprehessian = PriorPrecHessian(Jhat_m, reg, ka_opt)

        ## Number of components, number of iteration, and randomized SVD
        n_components = args.number_of_components
        n_iter = args.number_of_iterations
        n_extra = args.number_of_extra_vectors
        U, Sigma, VT = randomized_svd1(priorprehessian, n_components= n_components, n_iter= n_iter, n_oversamples = n_extra, size = len(ka_opt.vector()[:]))

        #np.savetxt('opt.txt', ka_opt.vector()[:])
        #np.savetxt('U.txt', U)
        #np.savetxt('Sigma.txt', Sigma)
        #np.savetxt('VT.txt', VT)

    # TODO: options to write subspaces to file

    #########################################################################
    ## With U(VT), we can define the reduced space problem: #################
    ## Inverse problem with reduced space and full observation ##############
    #########################################################################

    ## Initializing the array to optimize
    intermediate = np.random.zeros(n_components)
    ai = Ndarray(intermediate.shape, buffer=intermediate)

    with get_working_tape().name_scope("reduced_parameters"):
        Uai = dot_to_function(disc.parameter_space, U, ai)
        
    with get_working_tape().name_scope("embedded_parameters"):
        ## Converting the array into function
        ka_new_opt = Function(disc.parameter_space)
        ka_new_opt.assign(ka_opt + Uai)
    

    if args.check_inverse_problem:

        ## Next we evaluate he misfit in the reduced space
        m_red = misfit(ka_new_opt)

        ## Making Jhat2
        Jhat2 = ReducedFunctional_(m_red, Control(ai))

        iters = args.number_of_iterations_for_model
        ## We want to compare the original regularized full parameter space solution
        # to the unregularized subspace solution

        problem1 = MinimizationProblem(Jhat2)#, constraints=ResidualConstraint(10, Jhat, U))
        parameters = {"acceptable_tol": 1.0e-3, "maximum_iterations": iters}
        solver1 = IPOPTSolver(problem1, parameters=parameters)

        ka_opt1 = solver1.solve()     
        
        ka_opt2 = ka_opt.copy(deepcopy = True)
        ka_opt2.vector()[:] += U.dot(ka_opt1)
        
        firstplot = plot(ka_opt2)
        plt.colorbar(firstplot, ticks = [-0.5, 0, 0.1, 0.25, 0.5, 1])
        plt.figure()
        secondplot = plot(ka_opt)
        plt.colorbar(secondplot, ticks = [-0.5, 0, 0.1, 0.25, 0.5, 1])  
        plt.figure()
        plt.plot(Sigma)
        plt.show()
    
    #########################################################################
    ## Finding the range of the pressure at specific point with full space###
    #########################################################################

    if not args.prediction:
        return

    switch = args.reduced_boundary
    ## Pressure at the 0.5, 0.8    
    pred = Prediction(args, disc, name="prediction")

    if switch:
        pred_val = pred(ka_opt)
        msft_val = misfit(ka_opt)
    else:
        pred_val = pred(ka_new_opt)
        msft_val = misfit(ka_new_opt)

    lamda = AdjFloat(1.e-6)

    with get_working_tape().name_scope("continuation_prediction"):
        obj_val = msft_val + lamda * pred_val

    if switch:
        J_pred = ReducedFunctional_(obj_val, Control(ka_opt))
    else:
        J_pred = ReducedFunctional_(obj_val, Control(ai))

    get_working_tape().visualise()

    while True:
        print(msft_val, pred_val, obj_val)
        with stop_annotating():
            problem_pred_low = MinimizationProblem(J_pred)
            parameters = {"acceptable_tol": 1.0e-3, "maximum_iterations": 10, "print_level" : 7}
            solver_pred_low = IPOPTSolver(problem_pred_low, parameters=parameters)
            lamda = AdjFloat(lamda * 2.)
            if switch:
                ka_pred_low = solver_pred_low.solve()
                ka_opt[:] = ka_pred_low[:]
            else:
                ai_pred_low = solver_pred_low.solve()
                ai[:] = ai_pred_low[:]
        if switch:
            ka_loop = Function(ka_opt.function_space())
            ka_loop.assign(ka_opt)
        else:
            ka_loop = dot_to_function(disc.parameter_space, U, ai) + ka_opt
        msft_val = misfit(ka_loop)
        pred_val = pred(ka_loop)
        obj_val = msft_val + lamda * pred_val
        print(msft_val, pred_val, obj_val)
        if switch:
            J_pred = ReducedFunctional_(obj_val, Control(ka_opt))
        else:
            J_pred = ReducedFunctional_(obj_val, Control(ai))
        import pdb
        pdb.set_trace()
        # TODO: save/write (msft_fal, pred_val, lamda) to file for plotting

        #########################################################################
        ## Finding the range of the pressure at specific point ##################
        ## With reduced space ###################################################
        #########################################################################

        # Constrained optimization doesn't work well right now
        #Jhat_cen_red(ai)
        #problem_pred_low_red = MinimizationProblem(Jhat_cen_red, constraints=ResidualConstraint(1., Jhat_m_red))
        #parameters = {"acceptable_tol": 1.0e-4, "maximum_iterations": 10, "print_level" : 7}
        #solver_pred_low_red = IPOPTSolver(problem_pred_low_red, parameters=parameters)
        #ka_pred_low_red = solver_pred_low_red.solve()
        #print(U.dot(ka_pred_low_red))
        #prediction_lower_bound_red = Jhat_cen_red(ka_pred_low_red)
        #misfit_lower_bound_red = Jhat_m_red(ka_pred_low_red)
        #print(prediction_lower_bound_red, misfit_lower_bound_red)

        #Jhat_cen_red(ai)
        #problem_pred_up_red = MaximizationProblem(Jhat_cen_red, constraints=ResidualConstraint(1., Jhat_m_red))
        #parameters = {"acceptable_tol": 1.0e-4, "maximum_iterations": 10, "print_level" : 7}
        #solver_pred_up_red = IPOPTSolver(problem_pred_up_red, parameters=parameters)
        #ka_pred_up_red = solver_pred_up_red.solve()
        #print(U.dot(ka_pred_up_red))
        #prediction_upper_bound_red = Jhat_cen_red(ka_pred_up_red)
        #misfit_upper_bound_red = Jhat_m_red(ka_pred_up_red)
        #print(prediction_upper_bound_red, misfit_upper_bound_red)

    #print(prediction_upper_bound)
    #print(prediction_lower_bound)
    #print(prediction_upper_bound_red)
    #print(prediction_lower_bound_red)


    ### Save the result using existing program tool.
    #ka_opt_red = ka_opt.copy(deepcopy = True)
    #
    ## ka_opt2.vector()[:] = reject_outlier(U.dot(ka_opt1))
    ##ka_opt2.vector()[:] = U.dot(ka_opt1)
    #ka_opt_red.vector()[:] = U.dot(ka_opt1)
    ## print("Norm %f", np.linalg.norm(U.dot(ka_opt1)))

    ### prediction
    #pre_obs = prediction.obs
    #pre_state = prediction.state
    #pre_w = pre_state.solve(ka=ka_opt_red)
    ## import pdb
    ## pdb.set_trace()
    #pre_u, pre_p = pre_w.split(deepcopy=True)

    #firstplot = plot(ka_opt_red)
    #plt.colorbar(firstplot, ticks = [-0.5, 0, 0.1, 0.25, 0.5, 1])
    #plt.figure()
    #secondplot = plot(ka_opt)
    #plt.colorbar(secondplot, ticks = [-0.5, 0, 0.1, 0.25, 0.5, 1])  
    #plt.figure()
    #plt.plot(Sigma)
    #plt.show()

    #np.savetxt('Sigma.txt', Sigma)

    # xdmf_filename = XDMFFile("reduced_solution.xdmf")
    # xdmf_filename.write(ka_opt2)
