from fenics import *
from fenics_adjoint import *
from pyadjoint.tape import stop_annotating, get_working_tape

from discretization import Discretization
from state import State
from observation import Observation

class Prediction(object):

    def __init__(self, args, disc, obs=None, state=None, name = "prediction"):
        self.name = name
        
        self.disc = disc
        if obs is None:
            self.obs = Observation(args, disc)
        else:
            self.obs = obs
        if state is None:
            self.state = State(args, disc)
        else:
            self.state = state

    def prediction_center(self, ka, x_0 = 0.5, y_0 = 0.8):
        w = self.state.solve(ka=ka)
        w = self.obs.apply(w)
        # import pdb
        # pdb.set_trace()
        with get_working_tape().name_scope(self.name + "_center"):
            sigma = 0.01
            q_expr = Expression("1/(2*pi*sigma)*exp(-(0.5/sigma)*((x[0]-x_0)*(x[0]-x_0)+(x[1]-y_0)*(x[1]-y_0)))", x_0 = x_0, y_0 = y_0, sigma = sigma, degree = 3)
            q_adjflt = assemble(q_expr*abs(w[1])*dx)
        return q_adjflt


    def prediction_boundaries(self, ka):
        n1 = FacetNormal(self.disc.mesh)
        myds = Measure('ds', domain=self.disc.mesh, subdomain_data=self.disc.boundaries)

        w = self.state.solve(ka=ka)
        J = assemble(inner(w, w)*myds(1))
        return J


    def prediction_op(self, J, m):
        Jhat = ReducedFunctional(J, m)
        compute_gradient(Jhat.functional, Jhat.controls[0])
        return Jhat


    def __call__(self, ka, x_0 = 0.5, y_0 = 0.8):
        return self.prediction_center(ka, x_0=x_0, y_0=y_0)

    
