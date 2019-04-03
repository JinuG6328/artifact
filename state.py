from fenics import *
from fenics_adjoint import *

from initialize import *
from discretization import Discretization

class State(object):

    def __init__(self, disc):
        self.disc = disc
    
        self.W, self.bcs = get_state_space(disc.mesh, disc.boundaries)
        self.w = get_state_variable(self.W)

        self.A = get_function_space(self.disc.mesh)
        self.V = Constant(0.5)
        self.ka = interpolate(self.V, self.A)

        # self.u, self.p = None, None
        # self.v, self.q = None, None
        # self.a = None
        # self.Amat, self.b = None, None

    def _get_residual_form(self, w):
        # maybe need this?
        pass

    def apply(self, w, ka=None):
        """ Compute the residual of the state equations for w """
        # if we have not assembled the residual form yet, assemble it
        # (cache for later use, maybe)
        # assemble the residual form for w
        pass

    def solve(self, ka=None):
        self.u, self.p = TrialFunctions(self.W)
        self.v, self.q = TestFunctions(self.W)
        self.a = (inner( alpha(ka) * self.u, self.v) + (div(self.v)*self.p) + (div(self.u)*self.q))*dx 

        self.n1 = FacetNormal(self.disc.mesh)
        self.myds = Measure('ds', domain=self.disc.mesh, subdomain_data=self.disc.boundaries)
        
        self.L1 = dot(self.v,self.n1)*Constant(-1.)*self.myds(1)
        self.L2 = dot(self.v,self.n1)*Constant(1.)*self.myds(2)

        self.Amat, self.b = assemble_system(self.a, self.L1 + self.L2, self.bcs)
        solve(self.Amat, self.w.vector(), self.b)
        return self.w

    def set_ka(self):
        pass

    def get_ka(self):
        pass

        
