from fenics import *
from fenics_adjoint import *
import sympy as sym
import moola

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
		bcu_inflow = DirichletBC(W.sub(0), Constant((0.0,0.0)), boundaries, 1)
		bcu_outflow = DirichletBC(W.sub(0), Constant((0.0,0.0)), boundaries, 3)
		bcu_01 = DirichletBC(W.sub(0), Constant((0.0,0.0)), boundaries, 2)
		bcu_02 = DirichletBC(W.sub(0), Constant((0.0,0.0)), boundaries, 4)
		bcs = (bcu_inflow, bcu_outflow, bcu_01, bcu_02)
	else:
		bcs = None

	return W, bcs

def get_state_variable(W):
	return Function(W)

def get_coefficient_space(mesh):
	K = FunctionSpace(mesh, 'DG', 0)
	return K

def get_function_space(mesh):
	A = FunctionSpace(mesh, 'CG', 1)
	return A

def get_initial_coefficients(K):
	mesh = K.mesh()
	x = interpolate(Expression("x[0]", degree = 1,), K)
	y = interpolate(Expression("x[1]", degree = 1,), K)

	k = Function(K)
	v2d = K.dofmap().dofs()

	#len(v2d)=128 in n = 8
	for d in v2d:
		xx = x.vector()[d]
		yy = y.vector()[d]
		if 0.25 < xx < 0.75 and 0.25 < yy < 0.75:
			k.vector()[d] = 0.01
		else:
			k.vector()[d] = 1.
	return k

def get_rhs_space(mesh):
	P = FunctionSpace(mesh, "DG", 0)
	return P

def get_initial_rhs(P):
	mesh = P.mesh()
	g = interpolate(Expression("x[0]+x[1]",degree =1), P, name = 'SourceTerm')
	return g

def alpha(ka):
	return ka

def forward_problem(ka):
	(u,p) = split(w)
	(v,q) = TestFunctions(W)
	g = Constant(0.01)
	F = (inner(alpha(ka) * u,v) + (div(v)*p) - (div(u)*q)- g*q)*dx
	solve(F==0, w, bcs, solver_parameters={"newton_solver": {"absolute_tolerance": 1.0e-7,
                                                              "maximum_iterations": 20}})	
	return w

mesh, boundaries = get_mesh(32)
W, bcs = get_state_space(mesh)
w = get_state_variable(W)
#K = get_coefficient_space(mesh)
#k = get_initial_coefficients(K)
A = get_function_space(mesh)
#P = get_rhs_space(mesh)
#g = get_initial_rhs(P)

V = Constant(0.5)
w1 = Expression("sin(pi*x[0])*sin(pi*x[1])", degree = 3)
d = Expression("w", w = w1, degree = 3)

if __name__ == "__main__":
	
	ka = interpolate(V, A, name="Control") # initial guess.
	w = forward_problem(ka) 
	(u,p) = split(w)

	controls = File("output/control_iterations_guess.pvd")
	ka_viz = Function(A, name="ControlVisualisation")
    
	def eval_cb(j, ka):
		ka_viz.assign(ka)
		controls << ka_viz
	
	J = Functional((0.5*inner(d-p, d-p))*dx + 0.5*inner(grad(ka),grad(ka))*dx)
	m = Control(ka)
	Jhat = ReducedFunctional(J, m, eval_cb_post=eval_cb)

	lb = 0.0
	ub = 1.0

	problem = MinimizationProblem(Jhat, bounds=(lb, ub))

	parameters = {"acceptable_tol": 1.0e-3, "maximum_iterations": 100}
	solver = IPOPTSolver(problem, parameters=parameters)
	ka_opt = solver.solve()

	xdmf_filename = XDMFFile("output/final_solution.xdmf")
	xdmf_filename.write(ka_opt)