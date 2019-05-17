from fenics import *
from fenics_adjoint import *
from pyadjoint.tape import get_working_tape

from initialize import *
from discretization import *

import numpy as np
import h5py

class Observation(object):

    def __init__(self, args, disc, name="observation"):
        self.args = args
        self.disc = disc
        self.name = name
        self.W = disc.parameter_space
        self.D = disc.observation_space
        self.observed = None
        self.noise = args.obs_noise
        self.obs_write_file = args.obs_write_file
        self.obs_read_file = args.obs_read_file
        if self.obs_read_file:
            o = Function(self.D)
            input_file = HDF5File(self.disc.mesh.mpi_comm(), args.obs_read_file, "r")
            input_file.read(o)
            input_file.close()
            self.set_observed(o)


    def add_args(parser):
        parser.add_argument("-a", "--obs-noise",      type=float, default=0.,    help="add noise with specific standard deviation")
        parser.add_argument("-r", "--obs-read-file",  type=str,   default=None, help="file to read observations from")
        parser.add_argument("-w", "--obs-write-file", type=str,   default=None, help="file to write observations to")

    
    def apply(self, w):
        """ Apply the observation operator to w to get
        a vector that lives in the observation space """

        # right now, for this example, this can just be a copy of w
        with get_working_tape().name_scope(self.name):
            wcopy = w.copy(deepcopy=True)
        return wcopy


    def __call__(self, w):
        return self.apply(w)

    def get_observed(self):
        return self.observed

    def set_observed(self,d_w,noise = None):
        """ store the observation that goes with the state d_w, adding appropriate noise """

        if noise is None:
            noise = self.noise

        with get_working_tape().name_scope(self.name + "_observed"):
            d = d_w.copy(deepcopy=True)
            size_d_n = d_w.vector().size()
            d.vector()[:] += np.random.normal(0, noise, size_d_n)
        self.observed = d
        
        if self.obs_write_file:
            output_file_mix = XDMFFile(self.disc.mesh.mpi_comm(), self.obs_write_file)
            output_file_mix.write(self.observed)
            output_file_mix.close()

