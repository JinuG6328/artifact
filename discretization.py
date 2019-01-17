
from fenics import *

import initialize

class Discretization(object):

    def __new__(self, namespace):
        self.n = None
        self.mesh = None
        self.boundaries = None
        self.state_space = None
        Discretization.__init__(self, namespace)
        return self
    
    def add_args(parser):
        parser.add_argument("-n", "--grid-n", type=int, default=32, help="number of cells per grid direction")

    def __init__(self, namespace):
        self.n = namespace.grid_n
        self.mesh, self.boundaries = initialize.get_mesh(self.n)
        self.state_space = initialize.get_state_space(self.mesh)
        
