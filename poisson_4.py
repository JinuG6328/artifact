from fenics import *
from fenics_adjoint import *
import sympy as sym
import moola
import numpy as np
import math

from scipy import linalg, sparse
from sklearn.utils import *
from sklearn.utils.extmath import svd_flip
from SVD import safe_sparse_dot, randomized_svd, randomized_range_finder
from Initialize import *

set_log_level(ERROR)

def forward_problem(ka):
	(u,p) = TrialFunctions(W)
	(v,q) = TestFunctions(W)
	a = (inner( alpha(ka) * u,v) + (div(v)*p) + (div(u)*q))*dx 
	n = FacetNormal(mesh)
	myds = Measure('ds', domain=mesh, subdomain_data=boundaries)
	L1 = dot(v,n)*Constant(-1.)*myds(1)
	L2 = dot(v,n)*Constant(1.)*myds(2)
	l = L1 + L2 
	solve(a==l, w, bcs)
	return w

if __name__ == "__main__":
    Size = 32
    mesh, boundaries = get_mesh(Size)
    W, bcs = get_state_space(mesh, boundaries)
    w = get_state_variable(W)
    A = get_function_space(mesh)
    V = Constant(0.5)
    Alpha = Constant(0.001)
    power = 1.

    d_p = Function(W.sub(1).collapse())
    input_file = HDF5File(mesh.mpi_comm(), "p.h5", "r")
    input_file.read(d_p, "Pressure")
    input_file.close()

    d_u = Function(W.sub(0).collapse())
    input_file = HDF5File(mesh.mpi_comm(), "u.h5", "r")
    input_file.read(d_u, "Velocity")
    input_file.close()

    ka = interpolate(V, A, name="Control") # initial guess.
    w = forward_problem(ka) 
    (u,p) = split(w)
    
    # (u,p) = w.split(True)
    # print(type(u))
    mesh1 = UnitSquareMesh(8,8)
    A1 = get_function_space(mesh1)
    #F1 = Function(A1)
    #F2 = Function(A1)
    #print(type(d_u))
    #lp = LagrangeInterpolator()
    #lp.interpolate(F1, d_p)
    d_p1 = project(d_p,A1)
    #print(type(F1))
    #import pdb
    #pdb.set_trace()
    p1= project(p,A1)
    # import pdb
    # pdb.set_trace()
    
    #lp.interpolate(F2, p)

    controls = File("output/control_iterations_guess_Alpha(%f)_p(%f).pvd" % (Alpha, power) )
    ka_viz = Function(A, name="ControlVisualisation")
    
    def eval_cb(j, ka):
        ka_viz.assign(ka)
        controls << ka_viz
	
        # TODO: see if we can construct a J consisting of a pressure at fixed number of evaluation points
    #J = Functional((0.5*inner(d_p-p, d_p-p)+0.5*inner(d_u-u, d_u-u))*dx + Alpha*(np.power(inner(grad(ka),grad(ka))+0.001,power))*dx)
	#J = Functional((0.5*inner(d_u-u, d_u-u))*dx + Alpha*(np.power(inner(grad(ka),grad(ka))+0.001,power))*dx)
    J = Functional((0.5*inner(d_p1-p1, d_p1-p1))*dx + Alpha*(np.power(inner(grad(ka),grad(ka))+0.001,power))*dx)
	#J = Functional((0.5*inner(d_p-p, d_p-p))*dx + Alpha*(np.power(inner(grad(ka),grad(ka))+0.001,power))*dx)

	#norm
    m = Control(ka)
    Jhat = ReducedFunctional(J, m, eval_cb_post=eval_cb)
	# H = hessian(J, m)
	# print(type(H))
    # n_components = 10
    # n_iter = 5
    #     # TODO: use A -- the function space of the parameter -- to get the size
    # U, Sigma, VT = randomized_svd(Jhat, n_components= n_components, n_iter= n_iter, size = (Size+1)*(Size+1)) # size should be the discrete vector size of q
    # # This if for RT
    # print(Sigma)
    lb = 0.0
    ub = 1.0
#224 the paper
    problem = MinimizationProblem(Jhat, bounds=(lb, ub))

    parameters = {"acceptable_tol": 1.0e-3, "maximum_iterations": 100}
    solver = IPOPTSolver(problem, parameters=parameters)
    ka_opt = solver.solve()

    xdmf_filename = XDMFFile("output/final_solution_Alpha(%f)_p(%f).xdmf" % (Alpha ,power))
    xdmf_filename.write(ka_opt)

    # (u1,p1) = w.split(True)
    # velocity_opt = File("opt_velocity.pvd")
    # V2 = W.sub(0).collapse()
    # velo_viz = Function(V2, name="velocity")
    # velo_viz.assign(u1)
    # velocity_opt << velo_viz


    # pressure_opt = File("opt_pressure.pvd")
    # V3 = W.sub(1).collapse()
    # pressure_viz = Function(V3, name="pressure")
    # pressure_viz.assign(p1)
    # pressure_opt << pressure_viz
