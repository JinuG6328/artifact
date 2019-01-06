from fenics import *
from fenics_adjoint import *
import sympy as sym
set_log_level(ERROR)

import moola

## Boundaries Class
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
    mesh = UnitSquareMesh(N,N)
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


def get_state_space(mesh,boundaries=None):
    ## Setting two variable
    V = FiniteElement("RT", mesh.ufl_cell(), 2)
    Q = FiniteElement("DG", mesh.ufl_cell(), 1)

    ## Make mixed space
    VQ = V*Q
    W = FunctionSpace(mesh, VQ)
    if boundaries:
        bcu_inflow  = DirichletBC(W.sub(0), 0.0, boundaries, 1)
        bcu_outflow = DirichletBC(W.sub(0), 0.0, boundaries, 3)
        bcu_01  = DirichletBC(W.sub(0), 0.0, boundaries, 2)
        bcu_02  = DirichletBC(W.sub(0), 0.0, boundaries, 4)
        bcs = (bcu_inflow,bcu_outflow,bcu_01,bcu_02)
    else:
        bcs = None
    return W,bcs

def get_state_variable(W):
    return Function(W)

def get_coefficient_space(mesh):
    K = FunctionSpace(mesh, 'DG', 0)
    return K

def get_initial_coefficients(K):
    mesh = K.mesh()
    x = interpolate(Expression("x[0]", degree=1), K)
    y = interpolate(Expression("x[1]", degree=1), K)

    k = Function(K)
    v2d = K.dofmap().dofs()
    
    #len(v2d) = 128
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
    g = interpolate(Expression("x[0]+x[1]", degree=1), P, name='Control2')
    return g

def get_forward_problem(W,w,k,g):
    (u,p) = split(w)
    (v,q) = TestFunctions(W)
    F = (inner(k * u, v) + (div(v)*p) - (div(u)*q) - g*q)*dx
    return F



def solver(N, mu = 1.33e-4, perm = 1e-16, rho = 2700, u_e = None):


    ## making mesh
    mesh = UnitSquareMesh(N,N)

    ## Setting two variable
    V = FiniteElement("RT", mesh.ufl_cell(), 2)
    Q = FiniteElement("DG", mesh.ufl_cell(), 1)

    ## Make mixed space
    VQ = V*Q
    W = FunctionSpace(mesh, VQ)

    #####################################################################  
    ## boundaries start
    #####################################################################

    ## boundaries = FacetFunction("size_t", mesh)
    boundaries = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
    boundaries.set_all(0)
    left.mark(boundaries, 1)
    right.mark(boundaries, 2)
    top.mark(boundaries, 3)
    bottom.mark(boundaries, 4)

    # bcu_inflow  = DirichletBC(W.sub(0), u_e, boundaries, 1)
    # bcu_outflow = DirichletBC(W.sub(0), u_e, boundaries, 3)
    # bcu_01  = DirichletBC(W.sub(0), u_e, boundaries, 2)
    # bcu_02  = DirichletBC(W.sub(0), u_e, boundaries, 4)

    bcu_inflow  = DirichletBC(W.sub(1), 0.0, boundaries, 1)
    bcu_outflow = DirichletBC(W.sub(1), 0.0, boundaries, 3)
    bcu_01  = DirichletBC(W.sub(1), 0.0, boundaries, 2)
    bcu_02  = DirichletBC(W.sub(1), 0.0, boundaries, 4)

    ## Velocity boudnary condition for inflow and outflow
    bcs = [bcu_inflow, bcu_outflow, bcu_01, bcu_02]

    #####################################################################  
    ## boundaries end
    #####################################################################

    #####################################################################  
    ## coefficient variable start
    #####################################################################

    K = FunctionSpace(mesh, 'DG', 0)

    x = interpolate(Expression("x[0]", degree=1), K)
    y = interpolate(Expression("x[1]", degree=1), K)

    k = Function(K)
    v2d = K.dofmap().dofs()
    
    #len(v2d) = 128
    for d in v2d:
        xx = x.vector()[d]
        yy = y.vector()[d]
        if xx < 0.5 and yy < 0.5:
            k.vector()[d] = 1.
        else:
            k.vector()[d] = 1.

    #####################################################################  
    ## coefficient variable end
    #####################################################################

    ## Define variational problems
    (u, p) = TrialFunctions(W)
    (v, q) = TestFunctions(W)

    ## Define constant
    mu = Constant(mu)
    rho = Constant(rho)
    perm = Constant(perm)
    p_D = Constant(0)

    ## Define new measures associated with exterior boundaries
    ds = Measure('ds', domain=mesh, subdomain_data=boundaries)

    ## define divergence by my self
    div_e = Expression('-2*pi*sin(pi*x[1])*sin(pi*x[0])', degree=2)
    n = FacetNormal(mesh)

    #####################################################################  
    ## Currnent version start
    #####################################################################   
    
    # define p space
    P_Space = FunctionSpace(mesh, "DG", 0)
    #f = interpolate(Expression("x[0]+x[1]", degree=1), P_Space, name='Control1')
    g = interpolate(Expression("x[0]+x[1]", degree=1), P_Space, name='Control2')
    print(type(P_Space))
    # k * dot(u, v) + p*div(v) - dot(v,n)*p_D*ds = f*v
    # div(u)* q = g * q
    
    ## Define varioantal problem form
    a = (k * dot(u,v) + p*div(v) + div(u)*q )*dx
    #a = (k * dot(u,v) - p*div(v) + div(u)*q)*dx 
    L1 = dot(v,n) * p_D * ds
    # print(shape(f))
    # print(shape(v))
    # print(shape(g))
    # print(shape(q))
    # print(shape(n))
    
    L2 = g * q * dx
    #L3 = f * v *dx

    #?# How can I add f*v 

    #L1 = dot(grad(q),n) * p_D * ds
    print(shape(grad(q)))
    #L3 = div_e * q * dx
    L = L1 + L2 #+ L3


    #####################################################################  
    ## Current version end
    #####################################################################   

    #####################################################################  
    ## Previous version start
    #####################################################################   

    # ## Define varioantal problem form
    # a = (k * dot(u,v) + p*div(v) + div(u)*q)*dx
    # #a = (k * dot(u,v) - p*div(v) + div(u)*q)*dx
    # L1 = dot(v,n) * p_D * ds(1)
    # L2 = dot(v,n) * p_D * ds(3)
    # L3 = div_e * q * dx
    # L = L1 + L2 + L3

    #####################################################################  
    ## Previous version end
    #####################################################################   

    ## Compute solution
    #w = Function(W)
    #solve(a == L, w, bcs)

    # Split the mixed solution using deepcopy
    (u, p) = w.split(True)    
    return W, w, g, a, L, bcs

def compute_errors(u_e, u, W, N = 8):
    """Compute various measures of the error u - u_e, where
    u is a finite element Function and u_e is an Expression."""

    # Get function space
    #print(W.num_sub_spaces())
    V0, V1 = W.split()
    #print(type(V0))
    
    # Explicit computation of L2 norm
    error = (u - u_e)**2*dx
    E1 = sqrt(abs(assemble(error)))

    # # Explicit interpolation of u_e onto the same space as u
    # mesh_1 = UnitSquareMesh(N,N)
    # V0 = FunctionSpace(mesh_1, "RT", 1)
    
    u_e_ = interpolate(u_e, V0.collapse())
    error = (u - u_e_)**2*dx
    E2 = sqrt(abs(assemble(error)))

    # # Explicit interpolation of u_e to higher-order elements.
    # # u will also be interpolated to the space Ve before integration
    # Ve = FunctionSpace(V0.mesh(), 'P', 5)
    # u_e_ = interpolate(u_e, Ve)
    # error = (u - u_e)**2*dx
    # E3 = sqrt(abs(assemble(error)))

    # # Infinity norm based on nodal values
    # u_e_ = interpolate(u_e, V0)
    # E4 = abs(u_e_.vector().array() - u.vector().array()).max()

    # # L2 norm
    # E5 = errornorm(u_e, u, norm_type='L2', degree_rise=3)

    # # H1 seminorm
    # E6 = errornorm(u_e, u, norm_type='H10', degree_rise=3)

    # # Collect error measures in a dictionary with self-explanatory keys
    # errors = {'u - u_e': E1,
    #           'u - interpolate(u_e, V)': E2,
    #           'interpolate(u, Ve) - interpolate(u_e, Ve)': E3,
    #           'infinity norm (of dofs)': E4,
    #           'L2 norm': E5,
    #           'H10 seminorm': E6}

    return E1

def compute_convergence_rates(u_e, max_degree=3, num_levels=5):
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
            errors = compute_errors(u_e, u, W, n)
            E[degree].append(errors)
            # print('2 x (%d x %d x %d) P%d mesh, %d unknowns, E1 = %g' %
            #   (n, n, n, degree, u.function_space().dim(), errors['u - u_e']))
            print(errors)
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




#u_e = Expression(('(1-1*x[1])','(1-1*x[0])'), degree=2)
u_e = Expression(('sin(pi*x[1])*cos(pi*x[0])','sin(pi*x[0])*cos(pi*x[1])'), degree=2)
mesh,boundaries = get_mesh(16)
W,bcs = get_state_space(mesh)
w = get_state_variable(W)
K = get_coefficient_space(mesh)
k = get_initial_coefficients(K)
P = get_rhs_space(mesh)
g = get_initial_rhs(P)
F = get_forward_problem(W,w,k,g)
solve(F==0,w,bcs)
#u, p, W, f, a, L, bcs = solver(8)



##########################################################################
### forward problem using dolfin
##########################################################################

w1 = Expression("sin(pi*x[0])*sin(pi*x[1])", degree=3)
d = 1/(pi)
d = Expression("d*w", d=d, w=w1, degree=3)

# print("check shape of p and d")
# print(shape(p))
# print(shape(d))

alpha = Constant(1e-6)
(u,p) = split(w)
J = Functional((0.5*inner(d-p, d-p))*dx + alpha/2*g**2*dx)
control = Control(g)

rf = ReducedFunctional(J, control)

problem = MoolaOptimizationProblem(rf)
f_moola = moola.DolfinPrimalVector(g)
solver = moola.NewtonCG(problem, f_moola, options={'gtol': 1e-9,
                                                   'maxiter': 20,
                                                   'display': 3,
                                                   'ncg_hesstol': 0})

sol = solver.solve()
f_opt = sol['control'].data

plot(f_opt, interactive=True, title="f_opt")
###################################################################
## editing
# Define the expressions of the analytical solution
f_analytic = Expression("2*pi*pi*w", w=w1, degree=3)
p_analytic = Expression("w1", w1=w1, degree=3)

# We can then compute the errors between numerical and analytical
# solutions.

g.assign(f_opt)
solve(F== 0, w, bcs)
(u,p) = w.split(deepcopy=True)
control_error = errornorm(f_analytic, f_opt,mesh=mesh)
state_error = errornorm(p_analytic, p, mesh=mesh)
print("h(min):           %e." % mesh.hmin())
print("Error in state:   %e." % state_error)
print("Error in control: %e." % control_error)


#print(compute_convergence_rates(u_e))
#u, p, W = solver(8, u_e=u_e)
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
print(p)