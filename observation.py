from fenics import *
from fenics_adjoint import *

from initialize import *
from discretization import *
from state import *

import numpy as np
import h5py

class Observation(object):

    def __init__(self, disc):
        self.disc = disc
        self.W = None # state space
        self.D = None # observation space
        self.noise_u = 0.0
        self.noise_p = 0.0
        self.noise_d = 0.0
        self.observed = None     # state_obs = State(disc)

    def add_args(parser):
        parser.add_argument("-a", "--add_noise", type=int, default=0, help="add noise with specific standard deviation")
    
    def apply(self, w):
        """ Apply the observation operator to w to get
        a vector that lives in the observation space """

        # right now, for this example, this can just be a copy of w
        return w.copy()

    def get_observed(self, W):
        #if self.observed is None:
        self.observed = Function(W)
        input_file = HDF5File(self.disc.mesh.mpi_comm(), "w.h5", "r")
        input_file.read(self.observed, "Mixed")
        input_file.close()
        #return self.observed

    def set_observed(self,d_w,noise_u=None, noise_p = None, noise_d = None):
        """ store the observation that goes with the state d_w, adding appropriate noise """

        if noise_u is None:
            noise_u = self.noise_u

        if noise_p is None:
            noise_p = self.noise_p

        if noise_d is None:
            noise_d = self.noise_d

        u_n, p_n = d_w.split(deepcopy=True)
        size_u_n = u_n.vector().size()//2
        size_p_n = p_n.vector().size()
        u_n.vector()[:] += np.random.multivariate_normal([0,0], [[noise_u,0],[0,noise_u]], size_u_n).flatten() 
        p_n.vector()[:] += np.random.normal(0, noise_p, size_p_n)
        assign(d_w.sub(1),p_n)
        assign(d_w.sub(0),u_n)

        d = self.apply(d_w)
        size_d_n = d.vector().size()
        d.vector()[:] += np.random.normal(0, noise_d, size_d_n)
        self.observed = d
        
        output_file_mix = HDF5File(self.disc.mesh.mpi_comm(), "w.h5", "w")
        output_file_mix.write(self.observed, "Mixed")
        output_file_mix.close()

        #self.noise = True
        # output_file_p = HDF5File(self.mesh.mpi_comm(), "p.h5", "w")
        # output_file_p.write(self.p_n, "Pressure")
        # output_file_p.close()

        # output_file_vel = HDF5File(self.mesh.mpi_comm(), "u.h5", "w")
        # output_file_vel.write(self.u_n, "Velocity")
        # output_file_vel.close()
