from dolfin import *
from dolfin_adjoint import *

try:
    import pyipopt
except ImportError:
    print("""This example depends on IPOPT and pyipopt. \
  When compiling IPOPT, make sure to link against HSL, as it \
  is a necessity for practical problems.""")
    raise

# turn off redundant output in parallel
parameters["std_out_all_processes"] = False

mu = Constant(1.0)                   # viscosity
alphaunderbar = 2.5 * mu / (100**2)  # parameter for \alpha
alphabar = 2.5 * mu / (0.01**2)      # parameter for \alpha
q = Constant(0.01) # q value that controls difficulty/discrete-valuedness of solution

def alpha(rho):
    """Inverse permeability as a function of rho, equation (40)"""
    return alphabar + (alphaunderbar - alphabar) * rho * (1 + q) / (rho + q)

N = 100
delta = 1.5  # The aspect ratio of the domain, 1 high and \delta wide
V = Constant(1.0/3) * delta  # want the fluid to occupy 1/3 of the domain

mesh = Mesh(RectangleMesh(MPI.comm_world, Point(0.0, 0.0), Point(delta, 1.0), N, N))
A = FunctionSpace(mesh, "CG", 1)        # control function space

U_h = VectorElement("CG", mesh.ufl_cell(), 2)
P_h = FiniteElement("CG", mesh.ufl_cell(), 1)
W = FunctionSpace(mesh, U_h*P_h)          # mixed Taylor-Hood function space

class InflowOutflow(UserExpression):
    def eval(self, values, x):
        values[1] = 0.0
        values[0] = 0.0
        l = 1.0/6.0
        gbar = 1.0

        if x[0] == 0.0 or x[0] == delta:
            if (1.0/4 - l/2) < x[1] < (1.0/4 + l/2):
                t = x[1] - 1.0/4
                values[0] = gbar*(1 - (2*t/l)**2)
            if (3.0/4 - l/2) < x[1] < (3.0/4 + l/2):
                t = x[1] - 3.0/4
                values[0] = gbar*(1 - (2*t/l)**2)

    def value_shape(self):
        return (2,)

def forward(rho):
    """Solve the forward problem for a given fluid distribution rho(x)."""
    w = Function(W)
    (u, p) = TrialFunctions(W)
    (v, q) = TestFunctions(W)

    F = (alpha(rho) * inner(u, v) * dx + inner(grad(u), grad(v)) * dx +
         inner(grad(p), v) * dx  + inner(div(u), q) * dx)
    bc = DirichletBC(W.sub(0), InflowOutflow(degree=1), "on_boundary")
    solve(lhs(F) == rhs(F), w, bcs=bc)

    return w

if __name__ == "__main__":
	rho = interpolate(Constant(float(V)/delta), A)
	w   = forward(rho)
	(u, p) = split(w)

	controls = File("output/control_iterations_guess.pvd")
	allctrls = File("output/allcontrols.pvd")
	rho_viz = Function(A, name="ControlVisualisation")

	def eval_cb(j, rho):
	    rho_viz.assign(rho)
	    controls << rho_viz
	    allctrls << rho_viz

	J = assemble(0.5 * inner(alpha(rho) * u, u) * dx + mu * inner(grad(u), grad(u)) * dx)
	m = Control(rho)
	Jhat = ReducedFunctional(J, m, eval_cb_post=eval_cb)

	# Bound constraints
	lb = 0.0
	ub = 1.0

	# We want V - \int rho dx >= 0, so write this as \int V/delta - rho dx >= 0
	volume_constraint = UFLInequalityConstraint((V/delta - rho)*dx, m)

	# Solve the optimisation problem with q = 0.01
	problem = MinimizationProblem(Jhat, bounds=(lb, ub), constraints=volume_constraint)
	parameters = {'maximum_iterations': 20}

	solver = IPOPTSolver(problem, parameters=parameters)
	rho_opt = solver.solve()

	rho_opt_xdmf = XDMFFile(MPI.comm_world, "output/control_solution_guess.xdmf")
	rho_opt_xdmf.write(rho_opt)
	
	q.assign(0.1)
	rho.assign(rho_opt)
	set_working_tape(Tape())
	
	rho_intrm = XDMFFile(MPI.comm_world, "intermediate-guess-%s.xdmf" % N)
	rho_intrm.write(rho)

	w = forward(rho)
	(u, p) = split(w)

	# Define the reduced functionals
	controls = File("output/control_iterations_final.pvd")
	rho_viz = Function(A, name="ControlVisualisation")
	
	def eval_cb(j, rho):
	    rho_viz.assign(rho)
	    controls << rho_viz
	    allctrls << rho_viz

	J = assemble(0.5 * inner(alpha(rho) * u, u) * dx + mu * inner(grad(u), grad(u)) * dx)
	m = Control(rho)
	Jhat = ReducedFunctional(J, m, eval_cb_post=eval_cb)

	problem = MinimizationProblem(Jhat, bounds=(lb, ub), constraints=volume_constraint)
	parameters = {'maximum_iterations': 100}

	solver = IPOPTSolver(problem, parameters=parameters)
	rho_opt = solver.solve()

	rho_opt_final = XDMFFile(MPI.comm_world, "output/control_solution_final.xdmf")
	rho_opt_final.write(rho_opt)