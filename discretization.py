
from fenics import *

import initialize

class Discretization(object):
    
    def __init__(self, namespace):
        self.n = namespace.grid_n
        self.mesh, self.boundaries = initialize.get_mesh(self.n)
        self.state_space, self.bcs = initialize.get_state_space(self.mesh)
        self.parameter_space = initialize.get_parameter_space(self.mesh)
        self.observation_space = self.parameter_space
    

    def add_args(parser):
        parser.add_argument("-n", "--grid-n", type=int, default=32, help="number of cells per grid direction")

        
