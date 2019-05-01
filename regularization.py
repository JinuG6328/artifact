from fenics import *
from fenics_adjoint import *
import ufl

from initialize import *
from discretization import Discretization
from state import *
# from SVD import *
import numpy as np

class Regularization(object):   
    def __init__(self, k, Functionspace, namespace):
        self.Alpha = namespace.regularization_alpha
        self.power = namespace.regularization_degree
        self.Functionspace = Functionspace

        self.reg_form = self.Alpha*(np.power(inner(grad(k),grad(k))+0.0001,self.power))*dx
        self.reg = assemble(self.reg_form) # float / adjointfloat (depending whether we're taping)

    def add_args(parser):
        parser.add_argument("-rd", "--regularization-degree", type=int, default=1, help="degree of regularization term")
        parser.add_argument("-ra", "--regularization-alpha", type=int, default=0.1, help="alpha value of regularization term")

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

    def __call__(self, input_):
        if isinstance(input_, Function):
            return self.reg
        elif isinstance(input_, tuple):
            return self.reg


