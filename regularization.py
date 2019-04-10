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
