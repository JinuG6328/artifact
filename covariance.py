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
        # y_hat = TestFunction(self._reg.Functionspace)
        hessian = self._rf.hessian(b)
        # y = dot(self.ka,hessian)*dx
        
        # coeffs = y.coefficients()
        # orig_y = coeffs[0]
        # replace_map = {}
        
        # ## Question about this part
        # replace_map[orig_y] = self.ka
        # new_form = ufl.replace(y, replace_map)
        # grad_form = ufl.derivative(new_form, self.ka, y_hat)
        
        # b2 = assemble(grad_form)
        
        # # get L from Regularization.compute_hessian() somehow
        L = self._reg.compute_hessian(self.ka) 
        z = self.ka.copy(deepcopy=True)
        # z1 = self.ka.copy(deepcopy=True)        
        
        # import pdb
        # pdb.set_trace()
        
        # solver = LUSolver(L)
        # solver.solve(z.vector(),hessian.vector(), annotate = False)

        solve(L,z.vector(),hessian.vector(), annotate= False)

        # import pdb
        # pdb.set_trace()
        
        fs = self.ka.function_space()
        q_dot = Function(fs)
        r_dot = Function(fs)
        size_of_mat = len(q_dot.vector()[:])
        s = (size_of_mat,size_of_mat)
        A_np = np.zeros(s)
        for i in range(size_of_mat):
            #print(i)
            c_dot = np.zeros(size_of_mat)
            c_dot[i] = 1
            q_dot.vector()[:] = np.ascontiguousarray(c_dot)
            solve(L, r_dot.vector(), q_dot.vector(), annotate = False)
            A_np[:,i] = r_dot.vector()[:]

        # import pdb
        # pdb.set_trace()
        
        print(A_np)
        return z