from fenics import *
from fenics_adjoint import *
from scipy import linalg
from sklearn.utils import *
from sklearn.utils.extmath import svd_flip
import numpy as np
import ufl

class PriorPrecHessian():
    def __init__(self, reduced_functional, regularization, ka, transpose=False, p_A=None):
        self._rf = reduced_functional
        self._reg = regularization
        self.ka = ka
        self.transpose = transpose
        if p_A is None:
            L = regularization.compute_hessian(self.ka)
            A = L.array()
            self.p_A = np.linalg.pinv(A)
        else:
            self.p_A = p_A
        if not transpose:
            self.T = PriorPrecHessian(reduced_functional, regularization, ka, transpose=True, p_A=self.p_A)

    def T(self):
        return self.T
    
    def dot(self, b): 
        p_A = self.p_A
        z = self.ka.copy(deepcopy=True)
        if self.transpose:
            z1 = self.ka.copy(deepcopy=True)
            z1.vector()[:] = p_A.dot(b.vector()[:])
            z.vector()[:] = self._rf.hessian(z1).vector()[:]
            return z
        else:
            hessian = self._rf.hessian(b).vector()[:]
            z.vector()[:] = p_A.dot(hessian)
            return z
        
