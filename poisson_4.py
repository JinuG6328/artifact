from fenics import *
from fenics_adjoint import *
import sympy as sym
import moola
import numpy as np
import math
import os
import argparse
import matplotlib.pyplot as plt
import pdb

from scipy import linalg, sparse
from sklearn.utils import *
from sklearn.utils.extmath import svd_flip
from SVD import safe_sparse_dot, randomized_svd, randomized_range_finder
from Initialize import *
from pprint import pprint

def forward_problem(ka):#, W, bcs):
    (u,p) = TrialFunctions(W)
    (v,q) = TestFunctions(W)
    mesh = W.mesh()
    a = (inner(alpha(ka) * u,v) + (div(v)*p) + (div(u)*q))*dx 
    n = FacetNormal(mesh)
    myds = Measure('ds', domain=mesh, subdomain_data=boundaries)
    L1 = dot(v,n)*Constant(-1.)*myds(1)
    L2 = dot(v,n)*Constant(1.)*myds(2)
    l = L1 + L2 
    solve(a==l, w, bcs)
    return w

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    Size = 16
    mesh, boundaries = get_mesh(Size)
    W, bcs = get_state_space(mesh, boundaries)
    w = get_state_variable(W)
    A = get_function_space(mesh)
    V = Constant(0.5)
    Alpha = Constant(0.) #Constant(0.001)
    power = 1.

    d_p = Function(W.sub(1).collapse())
    input_file = HDF5File(mesh.mpi_comm(), "p.h5", "r")
    input_file.read(d_p, "Pressure")
    input_file.close()

    d_u = Function(W.sub(0).collapse())
    input_file = HDF5File(mesh.mpi_comm(), "u.h5", "r")
    input_file.read(d_u, "Velocity")
    input_file.close()

    d_W = Function(W)
    input_file = HDF5File(mesh.mpi_comm(), "w.h5", "r")
    input_file.read(d_W, "Mixed")
    input_file.close()

    #version check
    ka = interpolate(V, A) # initial guess.
    print("This is Ka", type(ka))
    print(type(W))
    w = forward_problem(ka)#, W, bcs) 
    # u = split(w)[0]
    # p = split(w)[1]
    # (u1,p1) = w.split(deepcopy=True)

    controls = File("output/control_iterations_guess_Alpha(%f)_p(%f).pvd" % (Alpha, power) )
    ka_viz = Function(A, name="ControlVisualisation")
    
    def eval_cb(j, ka):
        ka_viz.assign(ka)
        controls << ka_viz
	
    # TODO: see if we can construct a J consisting of a pressure at fixed number of evaluation points
    # J = Functional((0.5*inner(z_i, d_p-p)+0.5*inner(r_i, d_u-u))*dx + Alpha*(np.power(inner(grad(ka),grad(ka))+0.001,power))*dx)
    # J = assemble(Alpha*(np.power(inner(grad(ka),grad(ka))+0.001,power))*dx)
    
    # print(type(J))
    # size_obs = 4
    # size_obs = 1
    
    # (u_trial,p_trial) = TrialFunctions(W)   
    # diff_vec = Function(W.sub(1).collapse())    
    # diff_vec.vector()[:] = np.ascontiguousarray(p.vector()-d_p.vector())
    # p2= interpolate(p,diff_vec)

    Size1 = 4
    mesh1, boundaries1 = get_mesh(Size1)
    W1, bcs1 = get_state_space(mesh1, boundaries1)
   
    DW = Function(W1)
    SW = Function(W1)

    A1 = get_function_space(mesh1)
    K1 = Function(A1)
    K1.interpolate(ka)

    DW.interpolate(d_W)
    SW.interpolate(w)
    #pdb.set_trace()
    # u = split(SW)[0]
    # p = split(SW)[1]
    # for k in range(1,size_obs+1):
    #     for j in range(1,size_obs+1):
    #         E_i = Expression("sin(k*pi*x[0])*sin(j*pi*x[1])", k = k, j = j, degree = 3)
    #         e_i = interpolate(E_i,W.sub(1).collapse())
    #         #pdb.set_trace()
    #         J_i = assemble(0.5*inner(e_i,d_p-w[1])*dx)
    #         # J_i = J_i_mat.inner(w.vector() - d_W.vector())
    #         #J = J + J_i * J_i
    #         J = J_i

    #J = Functional(((0.5*inner(z_i, d_p-p))*dx)**2 + Alpha*(np.power(inner(grad(ka),grad(ka))+0.001,power))*dx)
	#J = Functional((0.5*inner(d_u-u, d_u-u))*dx + Alpha*(np.power(inner(grad(ka),grad(ka))+0.001,power))*dx)
    J = assemble((0.5*inner(DW[1]-SW[1], DW[1]-SW[1]))*dx + Alpha*(np.power(inner(grad(K1),grad(K1))+0.001,power))*dx)
    #J = assemble((0.5*inner(d_p-p, d_p-p))*dx + Alpha*(np.power(inner(grad(ka),grad(ka))+0.001,power))*dx)

	#norm
    m = Control(K1)
    Jhat = ReducedFunctional(J, m, eval_cb_post=eval_cb)
    # print(type(Jhat))
    print("evaluationing Jhat")
    print(Jhat(K1))
    #H = hessian(J, m)
    #print(type(H))
    n_components = 10
    n_iter = 5

    #print(type(Jhat))
    print("Factoring Hessian")
    U, Sigma, VT = randomized_svd(Jhat, n_components= n_components, n_iter= n_iter, size = (Size1+1)*(Size1+1)) # size should be the discrete vector size of q
    print(Sigma)
   
        # TODO: use A -- the function space of the parameter -- to get the size
    #U, Sigma, VT = randomized_svd(Jhat, n_components= n_components, n_iter= n_iter, size = (Size+1)*(Size+1)) # size should be the discrete vector size of q
    # # This if for RT
    # print(Sigma)
    lb = 0.0
    ub = 1.0
#224 the paper
    problem = MinimizationProblem(Jhat, bounds=(lb, ub))

    parameters = {"acceptable_tol": 1.0e-3, "maximum_iterations": 100}
    solver = IPOPTSolver(problem, parameters=parameters)
    ka_opt = solver.solve()

    # U, Sigma, VT = randomized_svd(Jhat, n_components= n_components, n_iter= n_iter, size = (Size+1)*(Size+1))
    # print(Sigma)

    plt.figure()
    plot(p, title='p')
    plt.show()
    pdb.set_trace()
    xdmf_filename = XDMFFile("output/final_solution_Alpha(%f)_p(%f).xdmf" % (Alpha ,power))
    xdmf_filename.write(ka_opt)

    


    # (u1,p1) = w.split(True)
    # velocity_opt = File("opt_velocity.pvd")
    # V2 = W.sub(0).collapse()
    # velo_viz = Function(V2, name="velocity")
    # velo_viz.assign(u1)
    # velocity_opt << velo_viz

#argparse
    # pressure_opt = File("opt_pressure.pvd")
    # V3 = W.sub(1).collapse()
    # pressure_viz = Function(V3, name="pressure")
    # pressure_viz.assign(p1)
    # pressure_opt << pressure_viz

##interpolation to check specific value of observation
 #(u1,p1) = w.split(True)
    #\(u,p) = w.split(True)
    # # print(type(u))
    # mesh1 = UnitSquareMesh(8,8)
    # A1 = get_function_space(mesh1)
    # # #F1 = Function(A1)
    # # #F2 = Function(A1)
    # # #print(type(d_u))
    # # #lp = LagrangeInterpolator()
    # # #lp.interpolate(F1, d_p)
    # # d_p1 = interpolate(d_p,A1)
    # print("Finish d_p1")
    # # #print(type(F1))
    # # #import pdb
    # # #pdb.set_trace()
    # p1= interpolate(p1,A1)
    # # # import pdb
    # # # pdb.set_trace()
    #lp.interpolate(F2, p)

    # import pdb
    #pdb.set_trace()

#make another function space to code it.    
    #p_array = p.vector()[:]
    #print(len(p_array))
    #dp_array = d_p.vector()[:] 
    #print(len(dp_array))
    #diff_vec.vector()[:] = np.ascontiguousarray(p_array-dp_array)
