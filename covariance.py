from fenics import *
from fenics_adjoint import *
from scipy import linalg
from sklearn.utils import *
from sklearn.utils.extmath import svd_flip
import numpy as np
import ufl

class PriorPrecHessian():
    def __init__(self, reduced_functional, regularization_form, ka):
        self._rf = reduced_functional
        self._reg = regularization_form
        self.ka = ka

    def dot(self, b):
        import pdb
        pdb.set_trace()
        
        asdf = self._rf.hessian(b)
        y_hat = TestFunctions(self._reg.Functionspace)
        y = self._rf.hessian(b)*dx
        
        coeffs = y.coefficients()
        orig_y = coeffs[0]
        replace_map = {}
        
        ## Question about this part
        replace_map[orig_y] = asdf
        new_form = ufl.replace(y, replace_map)
        b1 = assemble(new_form)
        #b2 = GenericVector(self._rf.hessian(b).vector()[:])
        #b2.array() = self._rf.hessian(b).vector()[:]
        #b = assemble(new_form)
        # get L from Regularization.compute_hessian() somehow
        
        L = self._reg.compute_hessian(self.ka) 
        z = self.ka.copy(deepcopy=True)
        solve(L,z.vector(),b1)
        return z