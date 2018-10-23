from fenics import *
from fenics_adjoint import *
import sympy as sym
import numpy as np
import moola
import h5py

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
		
		# bcu_inflow = DirichletBC(W.sub(1), Constant(1.0), boundaries, 1)
		# bcu_outflow = DirichletBC(W.sub(1), Constant(-1.0), boundaries, 3)
		# bcu_01 = DirichletBC(W.sub(1), Constant(0.0), boundaries, 2)
		# bcu_02 = DirichletBC(W.sub(1), Constant(0.0), boundaries, 3)

		#bcu_inflow = DirichletBC(W.sub(0), Constant((1.0,0.0)), boundaries, 1)
		#bcu_outflow = DirichletBC(W.sub(0), Constant((0.0,0.0)), boundaries, 3)
		bcu_01 = DirichletBC(W.sub(0), Constant((0.0,0.0)), boundaries, 3)
		bcu_02 = DirichletBC(W.sub(0), Constant((0.0,0.0)), boundaries, 4)
		#bcs = (bcu_inflow, bcu_outflow, bcu_01, bcu_02)
		bcs = [bcu_01, bcu_02]
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
			k.vector()[d] = 0.1
			#k.vector()[d] = 1.
		else:
			k.vector()[d] = 1.0
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

mesh, boundaries = get_mesh(32)
W, bcs = get_state_space(mesh, boundaries)
w = get_state_variable(W)
K = get_coefficient_space(mesh)
k = get_initial_coefficients(K)

(u,p) = TrialFunctions(W)
(v,q) = TestFunctions(W)
a = (inner( k * u,v) + (div(v)*p) + (div(u)*q))*dx #- g*q)*dx

n = FacetNormal(mesh)
myds = Measure('ds', domain=mesh, subdomain_data=boundaries)
L1 = dot(v,n)*Constant(-1.)* myds(1)
L2 = dot(v,n)*Constant(1.)*myds(2)
l = L1 + L2 

solve(a==l, w, bcs)

(u1,p1) = split(w)
(u,p) = w.split(True)
velocity = File("etc/velocity.pvd")
pressure = File("etc/pressure.pvd")

# print(type(p))
# print(type(p1))
# print(p.ufl_shape)
# print(p.ufl_index_dimensions)
# print(p.operands())

# gamma = 1.e-5
# noise = np.random.normal(0, gamma, p.shape())
# p += noise


V = W.sub(0).collapse()
v_viz = Function(V, name = "Velocity")
v_viz.assign(u)
velocity << v_viz

pressure << w.split()[1]
pressure << mesh

# print(type(u))
# print(p.str())
#print(p.ufl_evaluate())

output_file = HDF5File(mesh.mpi_comm(), "p.h5", "w")
output_file.write(p, "Pressure")
output_file.close()

output_file_vel = HDF5File(mesh.mpi_comm(), "u.h5", "w")
output_file_vel.write(u, "Velocity")
output_file_vel.close()

