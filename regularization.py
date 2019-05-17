from fenics import *
from fenics_adjoint import *
from pyadjoint.tape import get_working_tape
import ufl

from initialize import *
from discretization import Discretization
from state import *
# from SVD import *
import numpy as np

class Regularization(object):   
    def __init__(self, args, disc, name="regularization"):
        self.args = args
        self.name = name
        self.Alpha = args.reg_alpha
        self.power = args.reg_degree
        self.Functionspace = disc.parameter_space
        self.ka = Function(self.Functionspace)
        self.reg_form = self.Alpha*(np.power(inner(grad(self.ka),grad(self.ka))+0.0001,self.power))*dx
        k_hat = TestFunction(self.Functionspace)
        self.grad_form = ufl.derivative(self.reg_form, self.ka, k_hat)
        k_tilde = TrialFunction(self.Functionspace)
        self.hess_form = ufl.derivative(self.grad_form, self.ka, k_tilde)

    def add_args(parser):
        parser.add_argument("-rd", "--reg-degree", type=int,   default=1,   help="degree of regularization term")
        parser.add_argument("-ra", "--reg-alpha",  type=float, default=0.1, help="alpha value of regularization term")

    def compute_hessian(self, ka):
        new_form = ufl.replace(self.hess_form, { self.ka : ka } )
        mat = assemble(new_form)
        return mat

    def __call__(self, ka):
        new_form = ufl.replace(self.reg_form, { self.ka : ka } )
        with get_working_tape().name_scope(self.name):
            reg = assemble(new_form)
        return reg


