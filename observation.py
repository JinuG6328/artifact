
from fenics import *
from fenics_adjoint import *

from initialize import *
from discretization import *
from state import *

import numpy as np
import h5py

class Observation(object):

    def __new__(self, disc):
        self.W = None
        self.bcs = None
        self.w = None
        self.K = None
        self.u, self.p = None, None
        self.v, self.q = None, None
        self.a = None
        self.n1 = None
        self.myds = None
        self.L1 = None
        self.L2 = None
        self.Amat, self.b = None, None
        self.ka = None
        self.u_n, self.p_n = None, None
        self.size_u_n = None
        self.size_p_n = None
        self.sig_p = None
        self.cov_u = None
        self.noise_u = None
        self.noise_p = None
        self.noise_w = None
        self.noise = None
        Observation.__init__(self,disc)
        return self
        
    def __init__(self, disc):
        # self.W, self.bcs = get_state_space(self.mesh, self.boundaries)
        # self.w = get_state_variable(self.W)

        # self.K = get_coefficient_space(self.mesh)
        # self.k = get_initial_coefficients(self.K)
        
        # (self.u,self.p) = TrialFunctions(self.W)
        # (self.v,self.q) = TestFunctions(self.W)
        # self.a = (inner( self.k * self.u, self.v) + (div(self.v)*self.p) + (div(self.u)*self.q))*dx 
        # self.n1 = FacetNormal(self.mesh)
        # self.myds = Measure('ds', domain=self.mesh, subdomain_data=self.boundaries)
        # self.L1 = dot(self.v,self.n1)*Constant(-1.)*self.myds(1)
        # self.L2 = dot(self.v,self.n1)*Constant(1.)*self.myds(2)
        # self.Amat, self.b = assemble_system(self.a, self.L1 + self.L2, self.bcs)
        # solve(self.Amat, self.w.vector(), self.b)
        state_obs = State(disc, False)

        self.noise = True
        
        if self.noise:
            (self.u_n,self.p_n) = state_obs.w.split(deepcopy=True)
            self.size_u_n = int(self.u_n.vector().size()/2)
            self.size_p_n = self.p_n.vector().size()
            self.cov_u = 0.01
            self.noise_u = np.random.multivariate_normal([0,0], [[self.cov_u,0],[0,self.cov_u]], self.size_u_n).flatten()
            self.u_n.vector()[:] += self.noise_u

            self.sig_p = 0.01
            self.noise_p = np.random.normal(0, self.sig_p, self.size_p_n)
            self.p_n.vector()[:] += self.noise_p 
            assign(state_obs.w.sub(1),self.p_n)
            assign(state_obs.w.sub(0),self.u_n)

        output_file_mix = HDF5File(disc.mesh.mpi_comm(), "w.h5", "w")
        output_file_mix.write(state_obs.w, "Mixed")
        output_file_mix.close()

        # output_file_p = HDF5File(self.mesh.mpi_comm(), "p.h5", "w")
        # output_file_p.write(self.p_n, "Pressure")
        # output_file_p.close()

        # output_file_vel = HDF5File(self.mesh.mpi_comm(), "u.h5", "w")
        # output_file_vel.write(self.u_n, "Velocity")
        # output_file_vel.close()

