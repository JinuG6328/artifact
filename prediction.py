from fenics import *
from fenics_adjoint import *
from pyadjoint.tape import stop_annotating, get_working_tape

from discretization import Discretization
from state import State
from observation import Observation

import numpy as np

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

        with get_working_tape().name_scope(self.name + "_center"):
            sigma = 0.01
            q_expr = Expression("exp(-(0.5/sigma)*((x[0]-x_0)*(x[0]-x_0)+(x[1]-y_0)*(x[1]-y_0)))", x_0 = x_0, y_0 = y_0, sigma = sigma, degree = 3)
            q_adjflt = assemble(q_expr*w[2]*dx)
        return q_adjflt

    def prediction_point(self, ka, x_0 = 0.5, y_0 = 0.8):
        # calculate state variabe using a given model parameter 
        w = self.state.solve(ka=ka)
        w = self.obs.apply(w)

        with get_working_tape().name_scope(self.name + "_point"):
            # define the subdomain, which is near the specific point, and use it to calculate the pressure/velocity at that point.
            # this is required in order to tape the state variable to AdjFloat.
            prediction_point = CompiledSubDomain("(abs(x[0]-c) < 0.05) && (abs(x[1]-d) < 0.05)", c = x_0, d = y_0)
            points = MeshFunction('size_t', self.disc.mesh, 0)
            points.set_all(0)
            prediction_point.mark(points, 1)
            dp = Measure('vertex', subdomain_data=points)
            sigma = 0.0001
            q_expr = Expression("exp(-(0.5/sigma)*((x[0]-x_0)*(x[0]-x_0)+(x[1]-y_0)*(x[1]-y_0)))", x_0 = x_0, y_0 = y_0, sigma = sigma, degree = 2)
            # q_adjflt_split = assemble(w[2]*dx)
            # q_adjflt_ver = assemble(p_res*dp(1))
            q_adjflt = assemble(q_expr*w[2]*dx)*1/sigma/np.pi/2
            # q_adjflt_split1 = assemble(p_res*dx(1))
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
        return self.prediction_point(ka, x_0=x_0, y_0=y_0)

    
