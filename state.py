import ufl
from fenics import *
from fenics_adjoint import *
from pyadjoint.tape import get_working_tape, stop_annotating

from initialize import *
from discretization import Discretization

class State(object):

    def __init__(self, namespace, disc, name="state"):
        self.disc = disc
        self.W, self.bcs = disc.state_space, disc.bcs
        self.A = disc.parameter_space
        self.w = Function(self.W)
        self.name = name

        u, p = split(self.w)
        z = TestFunction(self.W)
        v, q = split(z)

        with stop_annotating():
            self.ka = self.default_parameters()
        n1 = FacetNormal(self.disc.mesh)
        myds = Measure('ds', domain=self.disc.mesh, subdomain_data=self.disc.boundaries)
        L1 = dot(v,n1)*Constant(-1.)*myds(1)
        L2 = dot(v,n1)*Constant(1.)*myds(2)
        self.form = (inner( alpha1(self.ka) * u, v) + (div(v)*p) + (div(u)*q))*dx - (L1 + L2)
        what = TrialFunction(self.W)
        self.Jform = derivative(self.form, self.w, what)

    def default_parameters(self):
        ka = Function(self.A)
        ka.assign(Constant(0.5))
        return ka #interpolate(Constant(0.5), self.A)

    def apply(self, w, ka=None):
        """ Compute the residual of the state equations for w """
        if ka is None:
            ka = self.ka
        new_form = ufl.replace(self.form, { self.ka: ka, self.w: w })
        with get_working_tape().name_scope(self.name + "_residual"):
            res = assemble(new_form)
            self.bcs.apply(res)
        return res

    def __call__(self, w, ka=None):
        self.apply(w,ka=ka)

    def solve(self, ka=None, w=None):

        if w is None:
            w = self.w
        if ka is None:
            ka = self.ka
        new_form = ufl.replace(self.form, { self.ka: ka, self.w: w })
        new_Jform = ufl.replace(self.Jform, { self.ka: ka, self.w: w })
        with get_working_tape().name_scope(self.name + "_solve"):
            prob = NonlinearVariationalProblem(new_form, w, self.bcs, new_Jform)
            solver = NonlinearVariationalSolver(prob)
            solver.solve()
        return w

    def set_ka(self, ka):
        self.ka = ka

    def get_ka(self):
        return self.ka

        
