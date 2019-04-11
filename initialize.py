from fenics import *
from fenics_adjoint import *
import numpy as np


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
    # import pdb
    # pdb.set_trace()
    V = FiniteElement("RT", mesh.ufl_cell(), 2)
    Q = FiniteElement("CG", mesh.ufl_cell(), 1)

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

def alpha1(ka):
    return exp(ka)

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
        if 0.25 < xx < 0.75 and 0.25 < yy < 0.75:
            k.vector()[d] = 0.1
        else:
            k.vector()[d] = 1.0
    return k
