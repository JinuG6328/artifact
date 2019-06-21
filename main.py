import tensorflow
import argparse
import os 

from fenics import *
from fenics_adjoint import *
import numpy as np
import matplotlib.pyplot as plt
import h5py
import pickle as pkl

from scipy import linalg
from sklearn.utils import *
from sklearn.utils.extmath import svd_flip
from pyadjoint.overloaded_function import overload_function
from pyadjoint.tape import stop_annotating, get_working_tape

from SVD_ import get_matrix, safe_sparse_dot, randomized_range_finder, randomized_svd1, reject_outlier
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
from log_overloaded_function import Log
from sqrt_overloaded_function import Sqrt
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
    parser.add_argument("-rb", "--reduced_boundary", action="store_false", help="pressure boundary from the reduced space otherwise that from full space")
    parser.add_argument("-so", "--save_optimal_solution", type=str, default=None, help="save optimal solution to .h5 file ")
    parser.add_argument("-lo", "--load_optimal_solution", type=str, default=None, help="load optimal solution from .h5 file")
    parser.add_argument("-ss", "--save_subspace", type=str, default=None, help="save subspaces to file")
    parser.add_argument("-ls", "--load_subspace", type=str, default=None, help="load subspaces from file")
    parser.add_argument("-ll", "--load_loop", type=str, default=None, help="load solution in the loop from file")
    parser.add_argument("-vf", "--verbosity-fenics", type=int, default=40, help="how verbose fenics output is (higher is quieter)")
    parser.add_argument("-vi", "--verbosity-ipopt", type=int, default=0, help="how verbose ipopt is (lower is quieter)")
    parser.add_argument("-m", "--matrix", action="store_true", help="using numpy matrix computation")

    ## We can change the settings in direcretization and   
    Discretization.add_args(parser)   
    Observation.add_args(parser)
    Regularization.add_args(parser)
    Misfit.add_args(parser)
  
    ## Getting input values
    args = parser.parse_args()

    set_log_level(args.verbosity_fenics)
    
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

    ## Next we combined misfit and regularization to define reduced functional objective
    m = misfit(ka)
    r = reg(ka)

    # print(m, r)

    with get_working_tape().name_scope("objective"):
        objective = m + r

    Jhat = ReducedFunctional_(objective, Control(ka))

    # Solving minimization problem and save the result
    if args.load_optimal_solution:
        ka_opt = state.default_parameters()
        optimal=HDF5File(disc.mesh.mpi_comm(), args.load_optimal_solution, "r")
        optimal.read(ka_opt, "Optimal_solution")
        optimal.close()

    else:
        with stop_annotating():
            problem = MinimizationProblem(Jhat)
            parameters = {"acceptable_tol": 1.0e-6, "maximum_iterations": 15, "print_level": args.verbosity_ipopt}
            solver = IPOPTSolver(problem, parameters=parameters)   
            opt_sol = solver.solve().copy(deepcopy=True)

        with get_working_tape().name_scope("optimal_parameters"):
            ka_opt = opt_sol.copy(deepcopy=True)

        if args.save_optimal_solution:
            optimal=XDMFFile(args.save_optimal_solution.replace('h5','xdmf')
            optimal.write(ka_opt)
            optimal.close()
            optimal1=HDF5File(disc.mesh.mpi_comm(), args.save_optimal_solution, "w")
            optimal1.write(ka_opt, "Optimal_solution")
            optimal1.close()

    with stop_annotating():
        #########################################################################
        ## 9 components observation and Randomized SVD ##########################
        #########################################################################

        Jhat_m = ReducedFunctional_(m, Control(ka))

        ## Calculating PriorPreconditionedHessian matrix of Jhat_m
        priorprehessian = PriorPrecHessian(Jhat_m, reg, ka_opt)

        ## Number of components, number of iteration, and randomized SVD
        n_components = args.number_of_components
        n_iter = args.number_of_iterations
        n_extra = args.number_of_extra_vectors

        if args.load_subspace:
            f = open(args.load_subspace, 'r')
            U = np.load(f)
            Sigma = np.load(f)
            VT = np.load(f)
            f.close()
        else:
            U, Sigma, VT = randomized_svd1(priorprehessian, n_components= n_components, n_iter= n_iter, n_oversamples = n_extra, size = len(ka_opt.vector().get_local()), matrix=args.matrix)

        if args.save_subspace:
            f = open(args.save_subspace, 'w')
            np.save(f,U)
            np.save(f,Sigma)
            np.save(f,VT)
            f.close()
    
    #########################################################################
    ## With U(VT), we can define the reduced space problem: #################
    ## Inverse problem with reduced space and full observation ##############
    #########################################################################

    ## Initializing the array to optimize
    intermediate = np.zeros(n_components)
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

        problem1 = MinimizationProblem(Jhat2)
        parameters = {"acceptable_tol": 1.0e-3, "maximum_iterations": iters, "print_level": args.verbosity_ipopt}
        solver1 = IPOPTSolver(problem1, parameters=parameters)

        ka_opt1 = solver1.solve()     
        
        ka_opt2 = ka_opt.copy(deepcopy = True)
        # ka_opt2.vector().add_local(U.dot(ka_opt1))
        ka_opt2.vector().set_local(U.dot(ka_opt1))

        firstplot = plot(ka_opt2)
        plt.colorbar(firstplot)
        #plt.colorbar(firstplot, ticks = [-0.01, 0, 0.005, 0.01])
        plt.figure()
        secondplot = plot(ka_opt)
        plt.colorbar(secondplot)#, ticks = [-0.5, 0, 0.1, 0.25, 0.5, 1])  
        plt.figure()
        plt.xticks(range(0, len(Sigma)))
        plt.plot(Sigma)
        mode = ka_opt.copy(deepcopy = True)
        for i in range(0, 5):
            plt.figure()
            mode.vector().set_local(U[:,i])
            plot(mode)
        plt.show()
    
    #########################################################################
    ## Finding the range of the pressure at specific point with full space###
    #########################################################################

    switch = args.reduced_boundary
    switch = True
    ## Pressure at the 0.5, 0.8    
    pred = Prediction(args, disc, name="prediction")
    
    ka_opt1 = ka_opt.copy(deepcopy = True)
    ka_new_opt1 = Function(disc.parameter_space)
    ka_new_opt1.assign(ka_opt1 + Uai)
    
    if switch:
        pred_val = pred(ka_opt1)
        msft_val = misfit(ka_opt1)
    else:
        pred_val = pred(ka_new_opt1)
        msft_val = misfit(ka_new_opt1)

    lamda = AdjFloat(1.e-3)
    epsilon = AdjFloat(0.1)
    with get_working_tape().name_scope("continuation_prediction"):
        obj_val = - Log(epsilon-msft_val) + lamda * pred_val

    if switch:
        J_pred = ReducedFunctional_(obj_val, Control(ka_opt1))
        file = open('minimization_f.txt','w') 
    else:
        J_pred = ReducedFunctional_(obj_val, Control(ai))
        file = open('minimization_r.txt','w') 

    # get_working_tape().visualise()

    i = 0
    if args.load_loop: 
        loop = np.load('loop_min_f.npy')
    else:
        loop =[]

    while msft_val < 1.e-6 or i == 0:
        if args.load_loop:
            with stop_annotating():
                ka_opt1.vector().set_local(loop[i])   
            ka_loop = Function(ka_opt1.function_space())
            ka_loop.assign(ka_opt1)
        else:
            with stop_annotating():
                problem_pred_low = MinimizationProblem(J_pred)
                parameters = {"acceptable_tol": 1.0e-3, "maximum_iterations": 10, "print_level" : args.verbosity_ipopt}
                solver_pred_low = IPOPTSolver(problem_pred_low, parameters=parameters)
                if switch:
                    ka_pred_low = solver_pred_low.solve()
                    ka_opt1.vector().set_local(ka_pred_low.vector().get_local())
                else:
                    ai_pred_low = solver_pred_low.solve()
                    ai[:] = ai_pred_low[:]      
            if switch:
                ka_loop = Function(ka_opt1.function_space())
                ka_loop.assign(ka_opt1)
            else:
                ka_loop = dot_to_function(disc.parameter_space, U, ai) + ka_opt
            loop.append(ka_loop.vector().get_local())

        msft_val = misfit(ka_loop)
        pred_val = pred(ka_loop)
        
        lamda = AdjFloat(lamda * 2.)
        
        obj_val = - Log(epsilon-msft_val) + lamda * pred_val
        print("msft_val, pred_val, obj_val, lamda")
        print(msft_val, pred_val, obj_val, lamda)
        file.write("%g %g %g %g\n" % (msft_val, pred_val, obj_val, lamda)) 
        if switch:
            J_pred = ReducedFunctional_(obj_val, Control(ka_opt1))
        else:
            if args.load_loop:
                J_pred = ReducedFunctional_(obj_val, Control(ka_opt1))    
            else:
                J_pred = ReducedFunctional_(obj_val, Control(ai))
        i = i+1
    file.close()
    
    if args.load_loop == False:
        loop_ = np.asarray(loop)
        np.save('loop_min_f.npy', loop_)     
    
    lower_figure = plt.figure()
    lower_limit = state.solve(ka = ka_loop)
    lower = plot(lower_limit[1])
    plt.colorbar(lower)
    pkl.dump(lower_figure,open('min_f.pickle','wb'))

    # # get_working_tape().visualise()
    
    ## Maximization 
    intermediate = np.zeros(n_components)
    ai = Ndarray(intermediate.shape, buffer=intermediate)
    Uai = dot_to_function(disc.parameter_space, U, ai)

    ka_opt2 = ka_opt.copy(deepcopy = True)
    ka_new_opt2 = Function(disc.parameter_space)
    ka_new_opt2.assign(ka_opt2 + Uai)
    
    if switch:
        pred_val = pred(ka_opt2)
        msft_val = misfit(ka_opt2)
    else:
        pred_val = pred(ka_new_opt2)
        msft_val = misfit(ka_new_opt2)

    lamda = AdjFloat(1.e-3)
    epsilon = AdjFloat(0.1)
    with get_working_tape().name_scope("continuation_prediction"):
        obj_val = - Log(epsilon-msft_val) - lamda * pred_val
        # obj_val = msft_val - lamda * pred_val

    if switch:
        J_pred = ReducedFunctional_(obj_val, Control(ka_opt2))
        file = open('maximization_f.txt','w') 
    else:
        J_pred = ReducedFunctional_(obj_val, Control(ai))
        file = open('maximization_r.txt','w') 

    i = 0
    if args.load_loop: 
        loop = np.load('loop_max_f.npy')
    else:
        loop =[]

    while msft_val < 1.e-6 or i == 0:
        if args.load_loop:
            with stop_annotating():
                ka_opt1.vector().set_local(loop[i])   
            ka_loop = Function(ka_opt2.function_space())
            ka_loop.assign(ka_opt2)
        else:
            with stop_annotating():
                problem_pred_up = MinimizationProblem(J_pred)
                parameters = {"acceptable_tol": 1.0e-3, "maximum_iterations": 10, "print_level" : args.verbosity_ipopt}
                solver_pred_up = IPOPTSolver(problem_pred_up, parameters=parameters)
                if switch:
                    ka_pred_up = solver_pred_up.solve()
                    ka_opt2.vector().set_local(ka_pred_up.vector().get_local())
                else:
                    ai_pred_up = solver_pred_up.solve()
                    ai[:] = ai_pred_up[:]      
            if switch:
                ka_loop = Function(ka_opt2.function_space())
                ka_loop.assign(ka_opt2)
            else:
                ka_loop = dot_to_function(disc.parameter_space, U, ai) + ka_opt
            loop.append(ka_loop.vector().get_local())

        msft_val = misfit(ka_loop)
        pred_val = pred(ka_loop)
        
        lamda = AdjFloat(lamda * 2.)
        
        obj_val = - Log(epsilon-msft_val) - lamda * pred_val
        # obj_val = msft_val - lamda * pred_val
        print("msft_val, pred_val, obj_val, lamda")
        print(msft_val, pred_val, obj_val, lamda)
        
        if switch:
            J_pred = ReducedFunctional_(obj_val, Control(ka_opt2))
        else:
            if args.load_loop:
                J_pred = ReducedFunctional_(obj_val, Control(ka_opt1))    
            else:
                J_pred = ReducedFunctional_(obj_val, Control(ai))
        file.write("%g %g %g %g\n" % (msft_val, pred_val, obj_val, lamda))
        i = i+1    
         
    file.close()

    if args.load_loop == False:
        loop_ = np.asarray(loop)
        np.save('loop_max_f.npy', loop_)

    upper_figure = plt.figure()
    uper_limit = state.solve(ka = ka_loop)
    uper = plot(uper_limit[1])
    plt.colorbar(uper)
    pkl.dump(upper_figure,open('max_f.pickle','wb'))
    upper_figure_v = plt.figure()
    uper_v = plot(uper_limit[0])
    plt.colorbar(uper_v)
    pkl.dump(upper_figure_v,open('max_f_v.pickle','wb'))
    # plt.show()
    
    switch = False
    ## Pressure at the 0.5, 0.8    
    pred = Prediction(args, disc, name="prediction")
    
    ka_opt1 = ka_opt.copy(deepcopy = True)
    ka_new_opt1 = Function(disc.parameter_space)
    ka_new_opt1.assign(ka_opt1 + Uai)
    
    if switch:
        pred_val = pred(ka_opt1)
        msft_val = misfit(ka_opt1)
    else:
        pred_val = pred(ka_new_opt1)
        msft_val = misfit(ka_new_opt1)

    lamda = AdjFloat(1.e-3)
    epsilon = AdjFloat(0.1)
    with get_working_tape().name_scope("continuation_prediction"):
        obj_val = - Log(epsilon-msft_val) + lamda * pred_val

    if switch:
        J_pred = ReducedFunctional_(obj_val, Control(ka_opt1))
        file = open('minimization_f.txt','w') 
    else:
        J_pred = ReducedFunctional_(obj_val, Control(ai))
        file = open('minimization_r.txt','w') 

    # get_working_tape().visualise()
   
    i = 0
    if args.load_loop: 
        loop = np.load('loop_min_r.npy')
    else:
        loop =[]

    while msft_val < 1.e-6 or i == 0:
        if args.load_loop:
            with stop_annotating():
                ka_opt1.vector().set_local(loop[i])   
            ka_loop = Function(ka_opt1.function_space())
            ka_loop.assign(ka_opt1)
        else:
            with stop_annotating():
                problem_pred_low = MinimizationProblem(J_pred)
                parameters = {"acceptable_tol": 1.0e-3, "maximum_iterations": 10, "print_level" : args.verbosity_ipopt}
                solver_pred_low = IPOPTSolver(problem_pred_low, parameters=parameters)
                if switch:
                    ka_pred_low = solver_pred_low.solve()
                    ka_opt1.vector().set_local(ka_pred_low.vector().get_local())
                else:
                    ai_pred_low = solver_pred_low.solve()
                    ai[:] = ai_pred_low[:]      
            if switch:
                ka_loop = Function(ka_opt1.function_space())
                ka_loop.assign(ka_opt1)
            else:
                ka_loop = Function(ka_opt1.function_space())
                ka_loop.assign(dot_to_function(disc.parameter_space, U, ai) + ka_opt1)
            loop.append(ka_loop.vector().get_local())

        msft_val = misfit(ka_loop)
        pred_val = pred(ka_loop)
        
        lamda = AdjFloat(lamda * 2.)
        
        obj_val = - Log(epsilon-msft_val) + lamda * pred_val
        print("msft_val, pred_val, obj_val, lamda")
        print(msft_val, pred_val, obj_val, lamda)
        file.write("%g %g %g %g\n" % (msft_val, pred_val, obj_val, lamda)) 
        if switch:
            J_pred = ReducedFunctional_(obj_val, Control(ka_opt1))
        else:
            if args.load_loop:
                J_pred = ReducedFunctional_(obj_val, Control(ka_opt1))    
            else:
                J_pred = ReducedFunctional_(obj_val, Control(ai))
        i = i+1  
    file.close()

    if args.load_loop == False:
        loop_ = np.asarray(loop)
        np.save('loop_min_r.npy', loop_)

    lower_figure_r = plt.figure()
    lower_limit = state.solve(ka = ka_loop)
    lower = plot(lower_limit[1])
    plt.colorbar(lower)
    pkl.dump(lower_figure_r,open('min_r.pickle','wb'))
    # get_working_tape().visualise()
    
    ## Maximization 
    intermediate = np.zeros(n_components)
    ai = Ndarray(intermediate.shape, buffer=intermediate)
    Uai = dot_to_function(disc.parameter_space, U, ai)

    ka_opt2 = ka_opt.copy(deepcopy = True)
    ka_new_opt2 = Function(disc.parameter_space)
    ka_new_opt2.assign(ka_opt2 + Uai)
    
    if switch:
        pred_val = pred(ka_opt2)
        msft_val = misfit(ka_opt2)
    else:
        pred_val = pred(ka_new_opt2)
        msft_val = misfit(ka_new_opt2)

    lamda = AdjFloat(1.e-3)
    epsilon = AdjFloat(0.1)
    with get_working_tape().name_scope("continuation_prediction"):
        obj_val = - Log(epsilon-msft_val) - lamda * pred_val
        # obj_val = msft_val - lamda * pred_val

    if switch:
        J_pred = ReducedFunctional_(obj_val, Control(ka_opt2))
        file = open('maximization_f.txt','w') 
    else:
        J_pred = ReducedFunctional_(obj_val, Control(ai))
        file = open('maximization_r.txt','w') 

    i = 0
    if args.load_loop: 
        loop = np.load('loop_max_r.npy')
    else:
        loop =[]

    while msft_val < 1.e-6 or i == 0:
        if args.load_loop:
            with stop_annotating():
                ka_opt2.vector().set_local(loop[i])   
            ka_loop = Function(ka_opt2.function_space())
            ka_loop.assign(ka_opt2)
        else:  
            with stop_annotating():
                problem_pred_up = MinimizationProblem(J_pred)
                parameters = {"acceptable_tol": 1.0e-3, "maximum_iterations": 10, "print_level" : args.verbosity_ipopt}
                solver_pred_up = IPOPTSolver(problem_pred_up, parameters=parameters)
                if switch:
                    ka_pred_up = solver_pred_up.solve()
                    ka_opt2.vector().set_local(ka_pred_up.vector().get_local())
                else:
                    ai_pred_up = solver_pred_up.solve()
                    ai[:] = ai_pred_up[:]      
            if switch:
                ka_loop = Function(ka_opt2.function_space())
                ka_loop.assign(ka_opt2)
            else:
                ka_loop = Function(ka_opt2.function_space())
                ka_loop.assign(dot_to_function(disc.parameter_space, U, ai) + ka_opt2)
            loop.append(ka_loop.vector().get_local())

        msft_val = misfit(ka_loop)
        pred_val = pred(ka_loop)
        
        lamda = AdjFloat(lamda * 2.)
        
        obj_val = - Log(epsilon-msft_val) - lamda * pred_val
        # obj_val = msft_val - lamda * pred_val
        print("msft_val, pred_val, obj_val, lamda")
        print(msft_val, pred_val, obj_val, lamda)
        
        if switch:
            J_pred = ReducedFunctional_(obj_val, Control(ka_opt2))
        else:
            if args.load_loop:
                J_pred = ReducedFunctional_(obj_val, Control(ka_opt2))    
            else:
                J_pred = ReducedFunctional_(obj_val, Control(ai))
        file.write("%g %g %g %g\n" % (msft_val, pred_val, obj_val, lamda))    
        i = i+1 
         
    file.close()

    if args.load_loop == False:
        loop_ = np.asarray(loop)
        np.save('loop_max_r.npy', loop_)

    upper_figure = plt.figure()
    uper_limit = state.solve(ka = ka_loop)
    uper = plot(uper_limit[1])
    plt.colorbar(uper)
    pkl.dump(upper_figure,open('max_r.pickle','wb'))
    upper_figure_v = plt.figure()
    uper1 = plot(uper_limit[0])
    plt.colorbar(uper1)
    pkl.dump(upper_figure_v ,open('max_r_v.pickle','wb'))
    plt.show()

