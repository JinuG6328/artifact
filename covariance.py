from fenics import *
from fenics_adjoint import *
from scipy import linalg
from sklearn.utils import *
from sklearn.utils.extmath import svd_flip
import numpy as np
import ufl

class PriorPrecHessian():
    def __init__(self, reduced_functional, regularization_form, ka, transpose=False):
        self._rf = reduced_functional
        self._reg = regularization_form
        self.ka = ka
        self.transpose = transpose
        if not transpose:
            self.T = PriorPrecHessian(reduced_functional, regularization_form, ka, transpose=True)

    def T(self):
        return self.T
    
    def dot(self, b): 
        
        if self.transpose:
            hessian = self._rf.hessian(b)
            # # get L from Regularization.compute_hessian() somehow
            # import pdb
            # pdb.set_trace()
            L = self._reg.compute_hessian(self.ka) 
            z = self.ka.copy(deepcopy=True)
            solve(L,z.vector(),hessian.vector(), annotate= False)
        else:
            hessian = self._rf.hessian(b)
            # # get L from Regularization.compute_hessian() somehow
            # import pdb
            # pdb.set_trace()
            L = self._reg.compute_hessian(self.ka) 
            z = self.ka.copy(deepcopy=True)
            solve(L,z.vector(),hessian.vector(), annotate= False)
            return z
        # import pdb
        # pdb.set_trace()
        
        # fs = self.ka.function_space()
        # q_dot = Function(fs)
        # r_dot = Function(fs)
        # size_of_mat = len(q_dot.vector()[:])
        # s = (size_of_mat,size_of_mat)
        # A_np = np.zeros(s)
        # for i in range(size_of_mat):
        #     #print(i)
        #     c_dot = np.zeros(size_of_mat)
        #     c_dot[i] = 1
        #     q_dot.vector()[:] = np.ascontiguousarray(c_dot)
        #     solve(L, r_dot.vector(), q_dot.vector(), annotate = False)
        #     A_np[:,i] = r_dot.vector()[:]

        # import pdb
        # pdb.set_trace()
        
        # print(A_np)
        