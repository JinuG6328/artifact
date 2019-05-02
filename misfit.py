from fenics import *
from fenics_adjoint import *
from pyadjoint.tape import stop_annotating

from discretization import Discretization
from state import State
from observation import Observation
# from SVD import *


import matplotlib.pyplot as plt

class Misfit(object):

    def __init__(self, args, disc, obs=None, state=None, name = None):

        self.n = args.num_components_misfit
        self.full = args.misfit_full_space
        
        self.disc = disc
        if obs is None:
            self.obs = Observation(args, disc)
        else:
            self.obs = obs
        if state is None:
            self.state = State(args, disc)
        else:
            self.state = state

        # Observation
        self.d_w = self.obs.get_observed()
        if self.d_w is None:
            with stop_annotating():
                ka = self.state.default_parameters()
                w = self.state.solve(ka = ka)
                self.obs.set_observed(self.obs.apply(w))
                self.d_w = self.obs.get_observed()


    def add_args(parser):
        parser.add_argument("-f", "--misfit_full_space", action="store_false", help="make misfit using full space")
        parser.add_argument("-nt", "--num_components_misfit", type=int, default=9, help="number of compoents in misfit (4, 9, 16, 25, ...))")

    
    def prediction_center(self, ka, x_0 = 0.5, y_0 = 0.8):
        w = self.state.solve(ka=ka)
        
        sigma = 0.1
        q_expr = Expression("exp(-(0.5/sigma)*((x[0]-x_0)*(x[0]-x_0)+(x[1]-y_0)*(x[1]-y_0)))", x_0 = x_0, y_0 = y_0, sigma = sigma, degree = 3)
        q_adjflt = assemble(q_expr*w[1]*dx)
    
        return q_adjflt


    def prediction_boundaries(self, ka):
        n1 = FacetNormal(self.disc.mesh)
        myds = Measure('ds', domain=self.disc.mesh, subdomain_data=self.disc.boundaries)

        w = self.state.solve(ka=ka)
        J = assemble(inner(w, w)*myds(1))
        return J


    def make_misfit_red(self, d_w, ka):
        w = self.state.solve(ka=ka)
        w = self.obs.apply(w)
        J = 0.
        l = int(sqrt(self.n))+1
        for i in range(1,l):
            for j in range(1,l):
                e = Expression("sin(i*pi * x[0]) * sin(j*pi * x[1])", degree = 9, i = i, j = j)
                mid = interpolate(e,self.state.W.sub(1).collapse())
                J_int = assemble((0.5*inner(w[1]-d_w[1], mid))*dx)
                J_int_2 = J_int*J_int
                J += J_int_2
        return J


    def make_misfit(self, d_w, ka):
        w = self.state.solve(ka=ka)
        w = self.obs.apply(w)
        J = assemble(0.5*inner(w[1]-d_w[1], w[1]-d_w[1])*dx )
        return J
        

    def misfit_op(self, J, m):
        import pdb
        pdb.set_trace()
        Jhat = ReducedFunctional(J, m)
        compute_gradient(Jhat.functional, Jhat.controls[0])
        return Jhat


    def __call__(self, ka, w_obs=None):
        ''' we want this to return the misfit between the observation process applied to w and the observed values '''
        if not w_obs:
            w_obs = self.d_w
        if self.full:
            return self.make_misfit_red(w_obs,ka)
        else:
            return self.make_misfit(w_obs,ka)

    
