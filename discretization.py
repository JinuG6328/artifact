
from fenics import *

import initialize

class Discretization(object):


    def __init__(self):
        self.n = None
        self.mesh = None
        self.boundaries = None
        self.state_space = None

    def add_args(parser):
        ''' Add options related to the mesh and discretization to the argument parser'''
        parser.add_argument("-n", "--grid-n", type=int, default=10, help="number of cells per grid direction")

    def __new__(self, namespace):
        n = namespace.grid_n
        self.mesh, self.boundaries = initialize.get_mesh(n)
        self.state_space = initialize.get_state_space(self.mesh)
        return self


