from fenics import *
from fenics_adjoint import *

from initialize import *
from discretization import Discretization
from state import *
from SVD import *
import numpy as np

class Regularization(object):   
    def __init__(self, k):
        self.Alpha = 0.1
        self.power = 1.0
        self.reg = assemble(self.Alpha*(np.power(inner(grad(k),grad(k))+0.0001,self.power))*dx)
        
    # def add_args(parser):
    #     ''' Add options related to the mesh and discretization to the argument parser'''
    #     parser.add_argument("-n", "--grid-n", type=int, default=32, help="number of cells per grid direction")