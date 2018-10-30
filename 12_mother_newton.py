from dolfin import *
from dolfin_adjoint import *

# Declare mesh and function spaces
mesh = UnitSquareMesh(32, 32)

V = FunctionSpace(mesh, "CG", 1) # state space
W = FunctionSpace(mesh, "DG", 0) # control space

z = interpolate(Constant(0), W) # zero initial guess
z.rename("Control", "Control")
u = Function(V, name='State')
v = TestFunction(V)

# Define and solve the Poisson equation
F = (inner(grad(u), grad(v)) - z*v)*dx
bc = DirichletBC(V, 0.0, "on_boundary")
solve(F == 0, u, bc)

# Turn off forward and adjoint solver output
set_log_level(ERROR)

# Define regularisation parameter
alpha = Constant(1e-6)

# Define observation data
x = SpatialCoordinate(mesh)
d = (1/(2*pi**2) + 2*alpha*pi**2)*sin(pi*x[0])*sin(pi*x[1])

# Define functional of interest and the reduced functional
J = Functional((0.5*inner(u-d, u-d))*dx + 0.5*alpha*z**2*dx)

control = Control(z)
rf = ReducedFunctional(J, control)
H = hessian(J, control)


q = interpolate(Expression("2*pi*pi*sin(pi*x[0])*sin(pi*x[1])", degree = 2), W)
u = interpolate(Expression("sin(pi*x[0])*sin(pi*x[1])", degree = 2), V)
#print(type(q))
K = inner(H(q),q)
print(type(K))
print(K)
#print(type(H(q)))
#print("Hi")
#print(H(q))
# z <- solution value 2*pi**2 sin(pi x) sin (pi y)
# u <- solution value of u is sin(pi x) sin (pi y) ? verify
# z^T H z = alpha \| z \|^2 - \|u \|^2 ?



# Define optimisation problem
problem = MinimizationProblem(rf)

# Now configure the optimisation solver
params_dict = {
    'General': {
        'Krylov': {
            'Type': 'Conjugate Gradients',
            'Absolute Tolerance': 1.0e-5,
            'Relative Tolerance': 1.0e-5,
            'Iteration Limit': 50,
        },
    },
    'Step': {
        'Type': 'Line Search',
        'Line Search': {
            'Descent Method': {
                'Type': 'Newton-Krylov'
            },
        }
    },
    'Status Test': {
        'Gradient Tolerance': 1e-8,
    }
}
parameters = {"acceptable_tol": 1.0e-3, "maximum_iterations": 100}
#solver = ROLSolver(problem, parameters = params_dict)
solver = IPOPTSolver(problem, parameters = parameters)
z_opt = solver.solve()

File("output/12/control.pvd") << z_opt
