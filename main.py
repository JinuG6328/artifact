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
    # parser.add_argument("-rb", "--reduced_boundary", action="store_false", help="pressure boundary from the reduced space otherwise that from full space")
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
            optimal=XDMFFile(args.save_optimal_solution+'.xdmf')
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
            [U, Sigma, VT] = np.load(args.load_subspace+".npy", allow_pickle=True)
        else:
            U, Sigma, VT = randomized_svd1(priorprehessian, n_components= n_components, n_iter= n_iter, n_oversamples = n_extra, size = len(ka_opt.vector().get_local()), matrix=args.matrix)

        if args.save_subspace:
            np.save(args.save_subspace,[U, Sigma, VT])
    
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
        ka_opt2.vector().set_local(U.dot(ka_opt1))

        firstplot = plot(ka_opt2)
        plt.colorbar(firstplot)
        plt.figure()
        secondplot = plot(ka_opt)
        plt.colorbar(secondplot)
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

    ## Pressure at the 0.5, 0.8    
    pred = Prediction(args, disc, name="prediction")

    name = ['min_f', 'max_f', 'min_r', 'max_r']
    name = ["%s_" %args.grid_n + i for i in name]

    for j in range(4):
        intermediate = np.zeros(n_components)
        ai = Ndarray(intermediate.shape, buffer=intermediate)
        Uai = dot_to_function(disc.parameter_space, U, ai)

        ka_opt1 = ka_opt.copy(deepcopy = True)
        ka_new_opt1 = Function(disc.parameter_space)
        ka_new_opt1.assign(ka_opt1 + Uai)
        switch = (j//2)^1

        if switch:
            pred_val = pred(ka_opt1)
            msft_val = misfit(ka_opt1)
        else:
            pred_val = pred(ka_new_opt1)
            msft_val = misfit(ka_new_opt1)

        lamda = AdjFloat(1.e-3)
        epsilon = AdjFloat(0.01)
        with get_working_tape().name_scope("continuation_prediction"):
            
            obj_val = - Log(epsilon-msft_val) +(-1)**j * lamda * pred_val

        if switch:
            J_pred = ReducedFunctional_(obj_val, Control(ka_opt1))
            file = open('%s.txt' %name[j] ,'w') 
        else:
            J_pred = ReducedFunctional_(obj_val, Control(ai))
            file = open('%s.txt' %name[j],'w') 

        i = 0
        # import pdb
        # pdb.set_trace()
        try: 
            filename = str(epsilon).replace('.','') + name[j]
            loop = np.load('%s.npy' %filename)
            args.load_loop = True
        except:
            loop =[]
            args.load_loop = False
        # loop =[]
        # args.load_loop = False  

        while msft_val < 1e-4 or i == 0:
            if args.load_loop:
                with stop_annotating():
                    # import pdb
                    # pdb.set_trace()
                    ka_opt1.vector().set_local(loop[i])   
                ka_loop = Function(ka_opt1.function_space())
                ka_loop.assign(ka_opt1)
            else:
                with stop_annotating():
                    problem_pred = MinimizationProblem(J_pred)
                    parameters = {"acceptable_tol": 1.0e-3, "maximum_iterations": 10, "print_level" : args.verbosity_ipopt}
                    solver_pred = IPOPTSolver(problem_pred, parameters=parameters)
                    if switch:
                        ka_pred = solver_pred.solve()
                        ka_opt1.vector().set_local(ka_pred.vector().get_local())
                    else:
                        ai_pred = solver_pred.solve()
                        ai[:] = ai_pred[:]      
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
            
            obj_val = - Log(epsilon-msft_val) + (-1)**j *lamda * pred_val
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
            filename1 = str(epsilon)[:1]+str(epsilon)[2:]+name[j]
            # import pdb
            # pdb.set_trace()
            np.save('%s.npy' %filename1, loop_)     
        
        figure = plt.figure()
        limit = state.solve(ka = ka_loop)
        # import pdb
        # pdb.set_trace()
        asdf, sdf = limit.split()
        lower = plot(sdf)
        plt.colorbar(lower)
        # figure1 = plt.figure()
        # lower = plot(limit[2])
        # plt.colorbar(lower)
        pkl.dump(figure,open('%s.pickle' %name[j],'wb'))

    plt.show()

