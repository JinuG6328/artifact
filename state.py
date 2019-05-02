import ufl
from fenics import *
from fenics_adjoint import *

from initialize import *
from discretization import Discretization

class State(object):

    def __init__(self, namespace, disc):
        self.disc = disc
    
        self.W, self.bcs = disc.state_space, disc.bcs

        self.A = disc.parameter_space

        self.w = Function(self.W)

        u, p = split(self.w)
        #u, p = self.w.split(False)
        #v, q = TestFunctions(self.W)
        z = TestFunction(self.W)
        v, q = split(z)

        self.ka = self.default_parameters()
        n1 = FacetNormal(self.disc.mesh)
        myds = Measure('ds', domain=self.disc.mesh, subdomain_data=self.disc.boundaries)
        L1 = dot(v,n1)*Constant(-1.)*myds(1)
        L2 = dot(v,n1)*Constant(1.)*myds(2)
        self.form = (inner( alpha1(self.ka) * u, v) + (div(v)*p) + (div(u)*q))*dx - (L1 + L2)
        what = TrialFunction(self.W)
        self.Jform = derivative(self.form, self.w, what)

    def default_parameters(self):
        return interpolate(Constant(0.5), self.A)

    def apply(self, w, ka=None):
        """ Compute the residual of the state equations for w """
        # if we have not assembled the residual form yet, assemble it
        # (cache for later use, maybe)
        # assemble the residual form for w
        pass

    def __call__(self, w, ka=None):

        if ka is None:
            ka = self.ka
        new_form = ufl.replace(self.form, { self.ka: ka, self.w: w })
        res = assemble(new_form)
        self.bcs.apply(res)
        return res

    def solve(self, ka=None, w=None):

        if w is None:
            w = self.w
        if ka is None:
            ka = self.ka
        new_form = ufl.replace(self.form, { self.ka: ka, self.w: w })
        new_Jform = ufl.replace(self.Jform, { self.ka: ka, self.w: w })
        prob = NonlinearVariationalProblem(new_form, w, self.bcs, new_Jform)
        solver = NonlinearVariationalSolver(prob)
        solver.solve()
        return w

    def set_ka(self, ka):
        self.ka = ka

    def get_ka(self):
        return self.ka

        
