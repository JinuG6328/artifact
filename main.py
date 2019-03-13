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
#from SVD import safe_sparse_dot, randomized_range_finder, randomized_svd
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
from numpy_block_var import Ndarray

## We already defined our customized function dot_to_function.
## Using this function, we could successfuly use adjoint equation to estimate the parameters. 
from dot_to_function import dot_to_function
dot_to_function = overload_function(dot_to_function, UpdatedBlock)

# import pdb
# pdb.set_trace()

## We can change the setting on the command line
parser = argparse.ArgumentParser()
parser.add_argument("-f", "--forward", action="store_true", help="solve the forward problem")
parser.add_argument("-i", "--inverse", action="store_true", help="solve the inverse problem")
parser.add_argument("-o", "--observation", action="store_true", help="make the observation")
parser.add_argument("-r", "--regularization", type=int, default=0, help="power of regularization term ")
parser.add_argument("-nc", "--number_of_components", type=int, default=10, help="number of components in Truncated SVD")

if __name__ == "__main__":

    set_log_level(16)
    
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
    reg = Regularization(state.ka)
    
    ## Next we combined misfit and regularization to define reduced functional objective
    objective = residual_red + reg.reg
    with tape.name_scope("misfit_first_part"):
        Jhat = misfit.misfit_op(objective, Control(state.ka))

    #Jhat_red = ReducedFunctional(residual_red, Control(state.))
    
    ######################################################
    ## Sovling minimization problem and save the result ##
    with tape.name_scope("minimization_first_part"):
        problem = MinimizationProblem(Jhat, bounds=(0.0, 1.0))
        parameters = {"acceptable_tol": 1.0e-3, "maximum_iterations": 1}
        solver = IPOPTSolver(problem, parameters=parameters)
        ka_opt = solver.solve()

    sol_residual = misfit.make_misfit(obs.observed, state.ka)
    # conv_rate = taylor_test(sol_residual, state.ka, state.ka*0.1)
    # import pdb
    # pdb.set_trace()
    # xdmf_filename = XDMFFile("output/final_solution_Alpha(%f)_p(%f).xdmf" % (Reg.Alpha, Reg.power))
    # xdmf_filename.write(ka_opt)
    ###################################################set_###

    # import pdb
    # pdb.set_trace()
    with tape.name_scope("misfit_at_solution"):
        sol_residual_red = misfit.make_misfit_red(obs.observed, ka_opt)

    tape.visualise()

    Jhat_red = ReducedFunctional(sol_residual_red, Control(ka_opt))

    hello = compute_gradient(Jhat_red.functional, Jhat_red.controls[0])

    ## At optimal point, we do partial SVD, get the vectors
    n_components = 20#100
    n_iter = 0#00   
    U, Sigma, VT = randomized_svd1(Jhat_red, n_components= n_components, n_iter= n_iter, size = (disc.n+1)*(disc.n+1))
    
    import pdb
    pdb.set_trace()
    ## Saving the U, Sigma, V^T
    # np.savetxt('U_cen.txt', U)
    # np.savetxt('Sigma_cen.txt', Sigma)
    # np.savetxt('VT_cen.txt', VT)
    
    # np.savetxt('U_no_reg.txt', U)
    # np.savetxt('Sigma_no_reg.txt', Sigma)set_lo
    # np.savetxt('VT_no_reg.txt', VT)
    
    ## Loading the U, Sigma, V^T
    # U = np.loadtxt('U_cen.txt')
    # Sigma = np.loadtxt('Sigma_cen.txt')
    # VT = np.loadtxt('VT_cen.txt')

    ## Make problem easier
    # U = U[:,1:3]
    # n_components = 2

    ## With vector, we can define the problem we're interested in:
    prediction = Misfit(args, disc, name="prediction")
    intermediate = np.random.rand(n_components)
    ai = Ndarray(intermediate.shape, buffer=intermediate)
        
    with tape.name_scope("putting_into_defined_function"):
        ka_new = dot_to_function(state.A, VT.T, ai)
        # ka_new_norm = assemble(dot(ka_new,ka_new)*dx)

    # intermediate1 = np.random.rand(n_components)
    # ai2 = Ndarray(intermediate1.shape, buffer=intermediate1)
    # ka_new1 = dot_to_function(state.A,U,ai2)
    # red_norm = ReducedFunctional(ka_new_norm, Control(ka_new))
    # conv_rate1 = taylor_test(red_norm, ka_new, ka_new1)

    # intermediate2 = np.random.rand(n_components)
    # ai3 = Ndarray(intermediate2.shape, buffer=intermediate2)
    # red_norm1 = ReducedFunctional(ka_new_norm, Control(ai))
    # print("ai3")
    # conv_rate2 = taylor_test(red_norm1, ai, ai3)

    with tape.name_scope("Making_residual"):
        residual2 = prediction.make_misfit(obs.observed,ka_new)

    # import pdb
    # pdb.set_trace()

    reg1 = Regularization(ka_new)
    reg2 = (ai.dot(ai)+0.001)

    objective2 = residual2 #+ reg1.reg
    Jhat2 = prediction.misfit_op(objective2, Control(ai))

    ## TODO constraints
    # constraints = UFLInequalityConstraint((V/delta - rho), ai)
    problem1 = MinimizationProblem(Jhat2)
    parameters = {"acceptable_tol": 1.0e-3, "maximum_iterations": 50}
    solver1 = IPOPTSolver(problem1, parameters=parameters)
    ka_opt1 = solver1.solve()     


    ## Taylor test
    h_input = np.random.rand(n_components)
    h = Ndarray(h_input.shape, buffer=h_input)
    print("Entire system")
    conv_rate = taylor_test(Jhat2, ai, h)
    
    ## Prediction
    # ai + Sigma[0]*V^T*error <= epsilon

    tape.visualise()

    # Save the result using existing program tool.
    ka_opt2 = ka_opt.copy(deepcopy = True)
    ka_opt2.vector()[:] = U.dot(ka_opt1)

    firstplot = plot(ka_opt2)
    plt.colorbar(firstplot, ticks = [-10, 0, 10, 50])
    plt.figure()
    secondplot = plot(ka_opt)
    plt.colorbar(secondplot, ticks = [0, 0.25, 0.5, 0.75, 1])  
    plt.show()


    import pdb
    pdb.set_trace()

    # xdmf_filename = XDMFFile("reduced_solution.xdmf")
    # xdmf_filename.write(ka_opt2)