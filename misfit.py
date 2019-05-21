from fenics import *
from fenics_adjoint import *
from pyadjoint.tape import stop_annotating, get_working_tape

from discretization import Discretization
from state import State
from observation import Observation
# from SVD import *


import matplotlib.pyplot as plt

class Misfit(object):

    def __init__(self, args, disc, obs=None, state=None, name = "misfit"):

        self.n = args.num_components_misfit
        self.full = args.misfit_full_space
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

        # Observation
        self.d_w = self.obs.get_observed()
        if self.d_w is None:
            with stop_annotating():
                ka = self.state.default_parameters()
                w = self.state.solve(ka = ka)
                d_w = self.obs.apply(w)
            self.obs.set_observed(d_w)
            self.d_w = self.obs.get_observed()


    def add_args(parser):
        parser.add_argument("-f", "--misfit_full_space", action="store_true", help="make misfit using full space")
        parser.add_argument("-nt", "--num_components_misfit", type=int, default=9, help="number of compoents in misfit (4, 9, 16, 25, ...))")

    
    def make_misfit_red(self, d_w, ka):
        w = self.state.solve(ka=ka)
        w = self.obs.apply(w)
        with get_working_tape().name_scope(self.name):
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
        with get_working_tape().name_scope(self.name):
            J = assemble(0.5*inner(w[1]-d_w[1], w[1]-d_w[1])*dx )
        return J
        

    def __call__(self, ka, w_obs=None):
        ''' we want this to return the misfit between the observation process applied to w and the observed values '''
        if not w_obs:
            w_obs = self.d_w
        if self.full:
            return self.make_misfit(w_obs,ka)
        else:
            return self.make_misfit_red(w_obs,ka)
    

def ReducedFunctional_(J, m):
    Jhat = ReducedFunctional(J, m)
    hello = compute_gradient(Jhat.functional, Jhat.controls[0])
    return Jhat
