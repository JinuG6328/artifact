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
from misfit import Misfit
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
    ## Inverse problem with full space and full observation #################
    #########################################################################

    ## Next we define regularization
    reg = Regularization(args, disc)

    ## get a parameter vector
    with get_working_tape().name_scope("parameters"):
        ka = state.default_parameters()

    ## Next we combined misfit and regularization to define reduced functional objective
    m = misfit(ka)
    r = reg(ka)

    with get_working_tape().name_scope("objective"):
        objective = m + r

    # objective = misfit(ka) + reg(ka)
    Jhat = misfit.misfit_op(objective, Control(ka))

    # Solving minimization problem and save the result
    with stop_annotating():
        problem = MinimizationProblem(Jhat)
        parameters = {"acceptable_tol": 1.0e-6, "maximum_iterations": 100}
        solver = IPOPTSolver(problem, parameters=parameters)   
        opt_sol = solver.solve().copy(deepcopy=True)

    with get_working_tape().name_scope("optimal_parameters"):
        ka_opt = opt_sol.copy(deepcopy=True)

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
        U, Sigma, VT = randomized_svd1(priorprehessian, n_components= n_components, n_iter= n_iter, size = (disc.n+1)*(disc.n+1))

    #########################################################################
    ## With U(VT), we can define the reduced space problem: #################
    ## Inverse problem with reduced space and full observation ##############
    #########################################################################

    ## Initializing the array to optimize
    intermediate = np.random.rand(n_components)
    ai = Ndarray(intermediate.shape, buffer=intermediate)

    with get_working_tape().name_scope("reduced_parameters"):
        Uai = dot_to_function(disc.parameter_space, U, ai)
        
    with get_working_tape().name_scope("embedded_parameters"):
        ## Converting the array into function
        ka_new_opt = Function(disc.parameter_space)
        ka_new_opt.assign(ka_opt + Uai)
    
    ## Next we evaluate he misfit in the reduced space
    m_red = misfit(ka_new_opt)
    
    ## Making Jhat_red
    Jhat_m_red = misfit.misfit_op(m_red, Control(ai))

    #########################################################################
    ## Finding the range of the pressure at specific point with full space###
    #########################################################################

    ## Pressure at the 0.5, 0.8    
    pred = Prediction(args, disc, name="prediction")
    pressure_cen = pred(ka)
    Jhat_cen = pred.prediction_op(pressure_cen, Control(ka))

    pressure_cen_red = pred(ka_new_opt)
    Jhat_cen_red = pred.prediction_op(pressure_cen_red, Control(ai))

    with stop_annotating():
        problem_pred_low = MinimizationProblem(Jhat_cen, constraints=ResidualConstraint(1, Jhat_m))
        parameters = {"acceptable_tol": 1.0e-3, "maximum_iterations": 10}
        solver_pred_low = IPOPTSolver(problem_pred_low, parameters=parameters)
        ka_pred_low = solver_pred_low.solve()

        problem_pred_up = MaximizationProblem(Jhat_cen, constraints=ResidualConstraint(1, Jhat_m))
        parameters = {"acceptable_tol": 1.0e-3, "maximum_iterations": 10}
        solver_pred_up = IPOPTSolver(problem_pred_up, parameters=parameters)
        ka_pred_up = solver_pred_up.solve()

        prediction_upper_bound = Jhat_cen(ka_pred_up)
        prediction_lower_bound = Jhat_cen(ka_pred_low)

        #########################################################################
        ## Finding the range of the pressure at specific point ##################
        ## With reduced space ###################################################
        #########################################################################

        problem_pred_low_red = MinimizationProblem(Jhat_cen_red, constraints=ResidualConstraint(1., Jhat_m_red))
        parameters = {"acceptable_tol": 1.0e-3, "maximum_iterations": 10}
        solver_pred_low_red = IPOPTSolver(problem_pred_low_red, parameters=parameters)
        ka_pred_low_red = solver_pred_low_red.solve()

        problem_pred_up_red = MaximizationProblem(Jhat_cen_red, constraints=ResidualConstraint(1., Jhat_m_red))
        parameters = {"acceptable_tol": 1.0e-3, "maximum_iterations": 10}
        solver_pred_up_red = IPOPTSolver(problem_pred_up_red, parameters=parameters)
        ka_pred_up_red = solver_pred_up_red.solve()

        prediction_upper_bound_red = Jhat_cen_red(ka_pred_up_red)
        prediction_lower_bound_red = Jhat_cen_red(ka_pred_low_red)

    print(prediction_upper_bound)
    print(prediction_lower_bound)
    print(prediction_upper_bound_red)
    print(prediction_lower_bound_red)
    import pdb
    pdb.set_trace()


    ## Save the result using existing program tool.
    ka_opt_red = ka_opt.copy(deepcopy = True)
    
    # ka_opt2.vector()[:] = reject_outlier(U.dot(ka_opt1))
    #ka_opt2.vector()[:] = U.dot(ka_opt1)
    ka_opt_red.vector()[:] = U.dot(ka_opt1)
    # print("Norm %f", np.linalg.norm(U.dot(ka_opt1)))

    ## prediction
    pre_obs = prediction.obs
    pre_state = prediction.state
    pre_w = pre_state.solve(ka=ka_opt_red)
    # import pdb
    # pdb.set_trace()
    pre_u, pre_p = pre_w.split(deepcopy=True)

    firstplot = plot(ka_opt_red)
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
