from fenics import *
from fenics_adjoint import *
import ufl

from initialize import *
from discretization import Discretization
from state import *
from SVD import *
import numpy as np

class Regularization(object):   
    def __init__(self, k, Functionspace, alpha = 0.1, power = 1.0):
        self.Alpha = alpha
        self.power = power
        self.Functionspace = Functionspace

        self.reg_form = self.Alpha*(np.power(inner(grad(k),grad(k))+0.0001,self.power))*dx
        self.reg = assemble(self.reg_form) # float / adjointfloat (depending whether we're taping)

    def compute_hessian(self, k):

        form = self.reg_form
        coeffs = form.coefficients() # list of Functions, should just contain original k
        orig_k = coeffs[0]
        replace_map = {}
        replace_map[orig_k] = k
        new_form = ufl.replace(form, replace_map)
        k_hat = TestFunction(self.Functionspace)
        grad_form = ufl.derivative(new_form, k, k_hat)
        k_tilde = TrialFunction(self.Functionspace)
        hess_form = ufl.derivative(grad_form, k, k_tilde)
        mat = assemble(hess_form)
                
        return mat
        
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
