from fenics import *

import numpy as np
import matplotlib.pyplot as plt

## Boundaries Class
class Left(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[0], 0.0)

class Right(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[0], 1.0)

class Bottom(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[2], 0.0)

class Top(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[2], 1.0)

class Front(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[1], 0.0)

class Back(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[1], 1.0)


def solver(N, mu = 1.33e-4, perm = 1e-16, rho = 2700, u_e = None):

    # Create mesh and define function space
    mesh = UnitCubeMesh(N, N, N)
    
    ## boundaries
    boundaries = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
    boundaries.set_all(0)
    left.mark(boundaries, 1)
    right.mark(boundaries, 2)
    top.mark(boundaries, 3)
    bottom.mark(boundaries, 4)
    front.mark(boundaries, 5)
    back.mark(boundaries, 6)

    ## Element
    V = FiniteElement("RT", mesh.ufl_cell(), 1)
    Q = FiniteElement("DG", mesh.ufl_cell(), 0)

    ## Make mixed space
    VQ = V*Q
    W = FunctionSpace(mesh, VQ)

    ## Boundary
    if (u_e):
        bcu_inflow = DirichletBC(W.sub(0), u_e, boundaries, 1)
        bcu_outflow = DirichletBC(W.sub(0), u_e, boundaries, 6)
        bcu_01  = DirichletBC(W.sub(0), u_e, boundaries, 2)
        bcu_02  = DirichletBC(W.sub(0), u_e, boundaries, 3)
        bcu_03  = DirichletBC(W.sub(0), u_e, boundaries, 4)
        bcu_04  = DirichletBC(W.sub(0), u_e, boundaries, 5)
    else:
        bcu_inflow  = DirichletBC(W.sub(0), Constant((1.0,0,0)), boundaries, 1)
        bcu_outflow = DirichletBC(W.sub(0), Constant((0,1.0,0)), boundaries, 6)
        bcu_01  = DirichletBC(W.sub(0), Constant((0,0,0)), boundaries, 2)
        bcu_02  = DirichletBC(W.sub(0), Constant((0,0,0)), boundaries, 3)
        bcu_03  = DirichletBC(W.sub(0), Constant((0,0,0)), boundaries, 4)
        bcu_04  = DirichletBC(W.sub(0), Constant((0,0,0)), boundaries, 5)

    ## Velocity boudnary condition for inflow and outflow
    bcs = [bcu_inflow, bcu_outflow, bcu_01, bcu_02, bcu_04]
    # bcs = [bcu_inflow, bcu_outflow]

    ## Define variational problems
    (u, p) = TrialFunctions(W)
    (v, q) = TestFunctions(W)

    ## Define constant
    mu = Constant(mu)
    rho = Constant(rho)
    perm = Constant(perm)
    g_in = Constant((0,0,0))
    g_out = Constant((0,1,0))

    ## Define new measures associated with exterior boundaries
    ds = Measure('ds', domain=mesh, subdomain_data=boundaries)

    ## Define varioantal problem form
    a = dot(u,v)*dx + perm/mu*p*div(v)*dx + div(u)*q*dx
    L = perm/mu*dot(v,g_in)*ds(1) #- perm/mu*dot(v,g_out)*ds(6)

    ## Compute solution
    w = Function(W)
    solve(a == L, w, bcs)

    # Split the mixed solution using deepcopy
    (u, p) = w.split(True)

    return u, p, W

def compute_errors(u_e, u, W):
    """Compute various measures of the error u - u_e, where
    u is a finite element Function and u_e is an Expression."""

    # Get function space
    #print(W.num_sub_spaces())
    V0, V1 = W.split()
    print(type(V0))
    
    # Explicit computation of L2 norm
    error = (u - u_e)**2*dx
    E1 = sqrt(abs(assemble(error)))

    # Explicit interpolation of u_e onto the same space as u
    u_e_ = interpolate(u_e, V0.collapse())
    error = (u - u_e_)**2*dx
    E2 = sqrt(abs(assemble(error)))

    # # Explicit interpolation of u_e to higher-order elements.
    # # u will also be interpolated to the space Ve before integration
    # Ve = FunctionSpace(V0.mesh(), 'P', 5)
    # u_e_ = interpolate(u_e, Ve)
    # error = (u - u_e)**2*dx
    # E3 = sqrt(abs(assemble(error)))

    # Infinity norm based on nodal values
    u_e_ = interpolate(u_e, V0.collapse())
    E3 = abs(u_e_.vector().array() - u.vector().array()).max()

    # L2 norm
    E4 = errornorm(u_e, u, norm_type='L2', degree_rise=3)

    # H1 seminorm
    E5 = errornorm(u_e, u, norm_type='H10', degree_rise=3)

    # Collect error measures in a dictionary with self-explanatory keys
    errors = {'u - u_e': E1,
              'u - interpolate(u_e, V)': E2,
              'infinity norm (of dofs)': E3,
              'L2 norm': E4,
              'H10 seminorm': E5}

    return E2

def compute_convergence_rates(u_e, max_degree=3, num_levels=3):
    "Compute convergences rates for various error norms"

    h = {}  # discretization parameter: h[degree][level]
    E = {}  # error measure(s): E[degree][level][error_type]

    # Iterate over degrees and mesh refinement levels
    degrees = range(1, max_degree + 1)
    for degree in degrees:
        n = 4  # coarsest mesh division
        h[degree] = []
        E[degree] = []
        for i in range(num_levels):
            h[degree].append(1.0 / n)
            u, p, W = solver(n, u_e = u_e)
            errors = compute_errors(u_e, u, W)
            E[degree].append(errors)
            print(' (%d x %d x %d) P%d mesh, %d unknowns, E1 = %g' %
              (n, n, n, degree, u.function_space().dim(), errors['u - u_e']))
            # print(errors)
            n *= 2

    # Compute convergence rates
    from math import log as ln  # log is a fenics name too
    etypes = list(E[1][0].keys())
    rates = {}
    for degree in degrees:
        rates[degree] = {}
        for error_type in sorted(etypes):
            rates[degree][error_type] = []
            for i in range(1, num_levels):
                Ei = E[degree][i][error_type]
                Eim1 = E[degree][i - 1][error_type]
                r = ln(Ei / Eim1) / ln(h[degree][i] / h[degree][i - 1])
                rates[degree][error_type].append(round(r, 2))

    return etypes, degrees, rates


## Initialize Sub Domain

left = Left() 
right = Right() 
top = Top() 
bottom = Bottom() 
front = Front() 
back = Back() 

u_e = Expression(('(1-1*x[1])*(1-x[2])','(1-1*x[0])*(1-x[2])','(1-x[0])*(1-x[1])'), degree=3)
u, p, W = solver(8, u_e=u_e)

#u_e = Expression(('(1-1*x[0])*(1-1*x[0])','(1-1*x[1])*(1-1*x[1])','0'), degree=3)

#a,b,c = compute_convergence_rates(u_e)
#print(a)
#print(b)
#print(c)

#error_L2 = errornorm(u_e, u, 'L2')
#print(error_L2)
## Plot Solution

#plot(mesh)
plot(u)
#plot(p)

# ## Save Solution
# vtkfile = File('poisson/solution1.pvd')
# vtkfile << u

# ## Compute error in L2 norm
# error_L2 = errornorm(u_D, u, 'L2')


# ## Compute the maximum error at vertices
# vertex_values_u_D = u_D.compute_vertex_values(mesh)
# vertex_values_u = u.compute_vertex_values(mesh)
# error_max = np.max(np.abs(vertex_values_u_D- vertex_values_u))


# ## Print error 
# print('error_L2 = ', error_L2)
# print('error_max = ', error_max)

# ## Holdprint
plt.show()
