from fenics import *
from fenics_adjoint import *

from initialize import *
from discretization import Discretization
from state import *
from SVD import *
import numpy as np

class Regularization(object):   
    def __init__(self, k, Functionspace, alpha = 0.1, power = 1.0):
        self.Alpha = alpha
        self.power = power
        
        # self.v, self.q = TestFunctions(Functionspace)
        # w = Function(Functionspace)
        # self.v1, self.q1 = split(w)
        self.reg_val = assemble(self.Alpha*np.power((inner(grad(k),grad(k))+0.0001),self.power)*dx)
        # import pdb
        # pdb.set_trace()
        
        import pdb
        pdb.set_trace()

        self.reg = assemble(self.Alpha*(np.power(inner(grad(k),grad(k))+0.0001,self.power))*dx)            
        ## Putting self.reg_val(Adjfloat) into gradient, but problem with mixedelement.
        # self.reg2 = assemble(grad(Constant(self.reg_val, cell = Functionspace.ufl_cell()))*dx)
        # self.reg1 = assemble(grad(self.Alpha*Constant(self.reg_val, cell = Functionspace.ufl_cell()))*dx)
        self.reg_func = assemble((self.Alpha*interpolate(Constant(self.reg_val), Functionspace))*dx)
        
    #     if isinstance(k, Function):
    #         import pdb
    #         pdb.set_trace()
    #         self.reg = assemble(self.Alpha*(np.power(inner(grad(k),grad(k))+0.0001,self.power))*dx)            
    #         ## Putting self.reg_val(Adjfloat) into gradient, but problem with mixedelement.
    #         # self.reg2 = assemble(grad(Constant(self.reg_val, cell = Functionspace.ufl_cell()))*dx)
    #         # self.reg1 = assemble(grad(self.Alpha*Constant(self.reg_val, cell = Functionspace.ufl_cell()))*dx)
    #         self.reg_func = assemble((self.Alpha*interpolate(Constant(self.reg_val), Functionspace))*dx)
            
    #     else:
    #         self.reg = assemble(self.Alpha*(np.power(np.gradient(k).dot(np.gradient(k))+0.0001,self.power))*dx)
    # # def add_args(parser):
    #     ''' Add options related to the mesh and discretization to the argument parser'''
    #     parser.add_argument("-n", "--grid-n", type=int, default=32, help="number of cells per grid direction")