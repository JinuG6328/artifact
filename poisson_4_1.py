from fenics import *
from fenics_adjoint import *
import sympy as sym
import moola
import numpy as np
import math

set_log_level(ERROR)

class Left(SubDomain):
	def inside(self, x, on_boundary):
		return near(x[0], 0.0)

class Right(SubDomain):
	def inside(self, x, on_boundary):
		return near(x[0], 1.0)

class Top(SubDomain):
	def inside(self, x, on_boundary):
		return near(x[1], 1.0)

class Bottom(SubDomain):
	def inside(self, x, on_boundary):
		return near(x[1], 0.0)

def get_mesh(N):

	#making mesh
	mesh = UnitSquareMesh(N,N)

	#making boundary
	left = Left()
	right = Right()
	top = Top()
	bottom = Bottom()
	boundaries = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
	boundaries.set_all(0)
	left.mark(boundaries, 1)
	right.mark(boundaries, 2)
	top.mark(boundaries, 3)
	bottom.mark(boundaries, 4)
	return mesh, boundaries

def get_state_space(mesh, boundaries=None):

	#setting two variable
	V = FiniteElement("RT", mesh.ufl_cell(), 2)
	Q = FiniteElement("DG", mesh.ufl_cell(), 1)

	#making mixed space
	VQ = V*Q
	W = FunctionSpace(mesh,VQ)

	if boundaries:
		
		bcu_01 = DirichletBC(W.sub(0), Constant((0.0,0.0)), boundaries, 3)
		bcu_02 = DirichletBC(W.sub(0), Constant((0.0,0.0)), boundaries, 4)
	
		bcs = [bcu_01, bcu_02]
	else:
		bcs = None

	return W, bcs

def get_state_variable(W):
	return Function(W)

def get_function_space(mesh):
	A = FunctionSpace(mesh, 'CG', 1)
	return A

def alpha(ka):
	return ka

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

mesh, boundaries = get_mesh(32)
W, bcs = get_state_space(mesh, boundaries)
w = get_state_variable(W)
A = get_function_space(mesh)
V = Constant(0.5)
Alpha = Constant(0.001)
power = 0.5

## Observed Data 
######################

d_p = Function(W.sub(1).collapse())
input_file = HDF5File(mesh.mpi_comm(), "p.h5", "r")
input_file.read(d_p, "Pressure")
input_file.close()

d_u = Function(W.sub(0).collapse())
input_file = HDF5File(mesh.mpi_comm(), "u.h5", "r")
input_file.read(d_u, "Velocity")
input_file.close()


# pressure_a = File("pressure.pvd")
# V1 = W.sub(1).collapse()
# dp_viz = Function(V1, name="pressure")
# dp_viz.assign(d_p)
# pressure_a << dp_viz

# velocity_a = File("velocity.pvd")
# V2 = W.sub(0).collapse()
# du_viz = Function(V2, name="Velocity")
# du_viz.assign(d_u)
# velocity_a << du_viz

########################
########################


if __name__ == "__main__":
	
	ka = interpolate(V, A, name="Control") # initial guess.
	w = forward_problem(ka) 
	(u,p) = w.split(True)

	controls = File("output/control_iterations_guess_Alpha(%f)_p(%d).pvd" % (Alpha, power) )
	ka_viz = Function(A, name="ControlVisualisation")
    
	def eval_cb(j, ka):
		ka_viz.assign(ka)
		controls << ka_viz
	
	J = Functional(0.5*inner(d_p-p, d_p-p)*dx + Alpha*(inner(grad(ka),grad(ka)) + 0.001)*dx)
	#J = Functional( 0.5*inner(d_u-u, d_u-u)*dx + Alpha*np.power((inner(grad(ka),grad(ka)) + 0.001),1.)*dx)
	#norm
	m = Control(ka)
	Jhat = ReducedFunctional(J, m, eval_cb_post=eval_cb)

	lb = 0.0
	ub = 1.0

	problem = MinimizationProblem(Jhat, bounds=(lb, ub))

	parameters = {"acceptable_tol": 1.0e-3, "maximum_iterations": 100}
	solver = IPOPTSolver(problem, parameters=parameters)
	ka_opt = solver.solve()

	#ka_opt = minimize(Jhat, method="CG", tol=1.0e-12, options={"gtol":1.0e-12})

	xdmf_filename = XDMFFile("output/final_solution_Alpha(%f)_p(%d).xdmf" % (Alpha ,power))
	xdmf_filename.write(ka_opt)

	velocity = File("opt_velocity.pvd")
	V2 = W.sub(0).collapse()
	velo_viz = Function(V2, name="velocity")
	velo_viz.assign(u)
	velocity << velo_viz


	dadada2 = File("opt_pressure.pvd")
	V3 = W.sub(1).collapse()
	pressure_viz = Function(V3, name="pressure")
	pressure_viz.assign(p)
	dadada2 << pressure_viz