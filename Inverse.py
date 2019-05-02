from fenics import *
from fenics_adjoint import *
import sympy as sym
#import moola
import numpy as np
import math

from scipy import linalg, sparse
from sklearn.utils import *
from sklearn.utils.extmath import svd_flip
from SVD import safe_sparse_dot, randomized_svd, randomized_range_finder
from initialize import *

def forward_problem(ka):
	(u,p) = TrialFunctions(W)
	(v,q) = TestFunctions(W)
	a = (inner( alpha(ka) * u,v) + (div(v)*p) + (div(u)*q))*dx 
	n = FacetNormal(mesh)
	myds = Measure('ds', domain=mesh, subdomain_data=boundaries)
	L1 = dot(v,n)*Constant(-1.)*myds(1)
	L2 = dot(v,n)*Constant(1.)*myds(2)
	A, b = assemble_system(a, L1 + L2, bcs)
	#F = a - L1 - L2
	solve(A, w.vector(), b)
	return w

def load_the_data(W):
	d_p = Function(W.sub(1).collapse())
	input_file = HDF5File(mesh.mpi_comm(), "p_n.h5", "r")
	input_file.read(d_p, "Pressure")
	input_file.close()

	d_u = Function(W.sub(0).collapse())
	input_file = HDF5File(mesh.mpi_comm(), "u_n.h5", "r")
	input_file.read(d_u, "Velocity")
	input_file.close()

	d_W = Function(W)
	input_file = HDF5File(mesh.mpi_comm(), "w.h5", "r")
	input_file.read(d_W, "Mixed")
	input_file.close()
	return d_p, d_u, d_W

def Inverse():
	Size = 32
	mesh, boundaries = get_mesh(Size)
	W, bcs = get_state_space(mesh, boundaries)
	w = get_state_variable(W)
	A = get_parameter_space(mesh)
	V = Constant(0.5)
	Alpha = Constant(0.0)
	power = 1.

	mesh1 = UnitSquareMesh(8,8)
	A1 = get_parameter_space(mesh1)
	F1 = Function(A1)
	
	ka = interpolate(V, A) # initial guess.
	w = forward_problem(ka) 
	(u,p) = split(w)

	controls = File("output/control_iterations_guess_Alpha(%f)_p(%f).pvd" % (Alpha, power) )
	ka_viz = Function(A, name="ControlVisualisation")
	
	def eval_cb(j, ka):
		ka_viz.assign(ka)
		controls << ka_viz
	
	#J = assemble((0.5*inner(DW[1]-SW[1], DW[1]-SW[1]))*dx)
	e = Expression("sin(pi * x[0]) * sin(pi * x[1])", degree = 1)
	f = interpolate(e,W.sub(1).collapse())
	J = assemble((0.5*inner(w[1]-d_W[1], f))*dx)
	J = J*J
	#J = J + assemble(Alpha*(np.power(inner(grad(ka),grad(ka))+0.001,power))*dx)
	#J = Functional((0.5*inner(d_p-p, d_p-p))*dx + Alpha*(np.power(inner(grad(ka),grad(ka))+0.001,power))*dx)

	#norm
	m = Control(ka)
	Jhat = ReducedFunctional(J, m, eval_cb_post=eval_cb)
	hello=compute_gradient(Jhat.functional, Jhat.controls[0])

	# H = hessian(J, m)
	# print(type(H))
	n_components = 3
	n_iter = 3
	#     # TODO: use A -- the function space of the parameter -- to get the size
	U, Sigma, VT = randomized_svd(Jhat, n_components= n_components, n_iter= n_iter, size = (Size+1)*(Size+1)) # size should be the discrete vector size of q
	# # This if for RT
	print(Sigma)
	lb = 0.0
	ub = 1.0
	
	#224 the paper
	problem = MinimizationProblem(Jhat, bounds=(lb, ub))

	parameters = {"acceptable_tol": 1.0e-3, "maximum_iterations": 100}
	solver = IPOPTSolver(problem, parameters=parameters)
	ka_opt = solver.solve()

	xdmf_filename = XDMFFile("output/final_solution_Alpha(%f)_p(%f).xdmf" % (Alpha ,power))
	xdmf_filename.write(ka_opt)
