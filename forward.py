
from fenics import *

import initialize
from discretization import Discretization

class Forward(Discretization):


    def __init__(self):
        Discretization.__init__(self)


    def add_args(parser):
        ''' Add options related to the mesh and discretization to the argument parser'''
        parser.add_argument("-n", "--grid-n", type=int, default=10, help="number of cells per grid direction")
        
    def __new__(self,namespace):
        Discretization.__new__(self,namespace)
        return self


