
from fenics import *
from fenics_adjoint import *

from initialize import *
from discretization import Discretization
import numpy as np
import h5py

class Observation(Discretization):

    def __new__(self, Discretization):
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
        self.A, self.b = None, None
        self.ka = None
        Observation.__init__(self)
        return self

    def __init__(self):
        self.W, self.bcs = get_state_space(self.mesh, self.boundaries)
        self.w = get_state_variable(self.W)
        self.K = get_coefficient_space(self.mesh)
        self.k = get_initial_coefficients(self.K)
        self.u, self.p = TrialFunction(self.W)
        self.v, self.q = TestFunction(self.W)
        self.a = (inner( alpha(self.ka) * self.u, self.v) + (div(self.v)*self.p) + (div(self.u)*self.q))*dx 
        self.n1 = FacetNormal(mesh)
        self.myds = Measure('ds', domain=self.mesh, subdomain_data=self.boundaries)
        self.L1 = dot(self.v,self.n1)*Constant(-1.)*self.myds(1)
        self.L2 = dot(self.v,self.n1)*Constant(1.)*self.myds(2)
        self.A, self.b = assemble_system(self.a, self.L1 + self.L2, self.bcs)
        solve(self.A, self.w.vector(), self.b)
    
        output_file_vel = HDF5File(self.mesh.mpi_comm(), "w.h5", "w")
        output_file_vel.write(self.w, "Mixed")
        output_file_vel.close()

        
