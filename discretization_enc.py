
from fenics import *

import initialize

class Discretization_Enc(object):

    # def __new__(self, namespace):
    #     self.n = None
    #     self.mesh = None
    #     self.boundaries = None
    #     self.state_space = None
    #     Discretization.__init__(self, namespace)
    #     return self
    
    def __init__(self, n_components):
        self.n = n_components
        self.mesh, self.boundaries = initialize.get_mesh(self.n)
        self.state_space = initialize.get_state_space(self.mesh)
        
