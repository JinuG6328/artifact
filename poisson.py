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
		bcu_inflow = DrichletBC(W.sub(0), 0.0, boundaries, 1)
		bcu_outflow = DrichletBC(W.sub(0), 0.0, boundaries, 3)
		bcu_01 = DrichletBC(W.sub(0), 0.0, boundaries, 2)
		bcu_02 = DrichletBC(W.sub(0), 0.0, boundaries, 4)
		bcs = (bcu_inflow, bcu_outflow, bcu_01, bcu_02)
	else:
		bcs = None

	return W, bcs

def get_state_variable(W):
	return Function(W)

def get_coefficient_space(mesh):
	K = FunctionSpace(mesh, 'DG', 0)
	return K

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
		if xx < 0.5 and yy < 0.5:
			k.vector()[d] = 1.
		else:
			k.vector()[d] = 1.
	return k

def get_rhs_space(mesh):
	P = FunctionSpace(mesh, "DG", 0)
	return P

def get_initial_rhs(P):
	mesh = P.mesh()
	g = interpolate(Expression("x[0]+x[1]",degree =1), P, name = 'Control')
	return g

def get_forward_problem(W, w, k, g):
	(u,p) = split(w)
	(v,q) = TestFunctions(W)
	F = (inner(k * u,v) + (div(v)*p) - (div(u)*q) - g*q)*dx
	return F

mesh, boundaries = get_mesh(32)
W, bcs = get_state_space(mesh)
w = get_state_variable(W)
K = get_coefficient_space(mesh)
k = get_initial_coefficients(K)
P = get_rhs_space(mesh)
g = get_initial_rhs(P)
F = get_forward_problem(W,w,k,g)
solve(F==0,w,bcs)

w1 = Expression("sin(pi*x[0])*sin(pi*x[1])", degree = 3)
d = Expression("w", w = w1, degree = 3)
# loop over d and add noise to values
# print(type(d))

# https://fenicsproject.org/qa/1484/adding-a-random-perturbation-to-a-solution-at-each-time-step/
# radius = interpolate(Expression("sqrt(x[0]*x[0]+x[1]*x[1]+x[2]*x[2])"), V)
# num_dofs_perturbed = radius.vector()[radius.vector()>0.5].size()
# u1_old.assign(u1)
# u1_old.vector()[radius.vector()>0.5] += 0.01*(0.5-random.random(num_dofs_perturbed))

alpha = Constant(1e-6)
(u, p) = split(w)

#RHS
#J = Functional((0.5*inner(d-p, d-p))*dx + alpha/2*g**2*dx)

#Coefficient
J = Functional((0.5*inner(d-p, d-p))*dx + alpha/2*g**2*dx)
control = Control(g)

rf = ReducedFunctional(J, control)

problem = MoolaOptimizationProblem(rf)
f_moola = moola.DolfinPrimalVector(g)
solver = moola.NewtonCG(problem, f_moola, options = {'gtol': 1e-9,
                                                   'maxiter': 20,
                                                   'display': 3,
                                                   'ncg_hesstol': 0})

sol = solver.solve()
f_opt = sol['control'].data

f_analytic = Expression("2*pi*pi*w", w = w1, degree = 3)
p_analytic = Expression("w1", w1 = w1, degree = 3)

g.assign(f_opt)
solve(F==0, w, bcs)
(u,p) = w.split(deepcopy=True)
control_error = errornorm(f_analytic, f_opt, mesh = mesh)
state_error = errornorm(p_analytic, p, mesh = mesh)
print("h(min):           %e." % mesh.hmin())
print("Error in state:   %e." % state_error)
print("Error in control: %e." % control_error)

#1) loop over d and add noise to the value.

# have different version of J
# Jrhs
# Jcoffeicient 
# regularization on the parameter. 
# optimization for parameter
# optimality condition of objective condition is value of state to parameter
# linear system of equation 