
from fenics import *
from fenics_adjoint import *

from initialize import *
from discretization import Discretization

class State(Discretization):

    def solve_state(self, ka):
        self.u, self.p = TrialFunctions(self.W)
        self.v, self.q = TestFunctions(self.W)
        w = Function(self.W)
        self.a = (inner( alpha(self.ka) * self.u, self.v) + (div(self.v)*self.p) + (div(self.u)*self.q))*dx 
        self.n1 = FacetNormal(self.disc.mesh)
        self.myds = Measure('ds', domain=self.disc.mesh, subdomain_data=self.boundaries)
        self.L1 = dot(self.v,self.n1)*Constant(-1.)*self.myds(1)
        self.L2 = dot(self.v,self.n1)*Constant(1.)*self.myds(2)
        self.Amat, self.b = assemble_system(self.a, self.L1 + self.L2, self.bcs)
        solve(self.Amat, w.vector(), self.b)
        return w

    def __new__(self, disc):
        self.disc = None
        self.W = None
        self.bcs = None
        self.w = None
        self.A = None
        self.V = None
        self.u, self.p = None, None
        self.v, self.q = None, None
        self.a = None
        self.n1 = None
        self.myds = None
        self.L1 = None
        self.L2 = None
        self.A, self.b = None, None
        self.ka = None
        State.__init__(self, disc)
        return self
        
    def __init__(self, disc):
        self.disc = disc
        self.W, self.bcs = get_state_space(disc.mesh, disc.boundaries)
        self.A = get_function_space(disc.mesh)
        self.V = Constant(0.5)
        self.Alpha = Constant(0.0)
        self.ka = interpolate(self.V, self.A)
        ka = self.ka
        self.w = State.solve_state(self, ka)

    


