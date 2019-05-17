from fenics import *
from fenics_adjoint import *
from numpy_block_var import Ndarray
import numpy as np

class ResidualConstraint(InequalityConstraint):
        """A class that enforces the residual constraint r(x) <= eps """

        def __init__(self, eps, rfn):
            self.eps = float(eps)
            self.rfn = rfn
            self.tmpvec = rfn.controls[0]._ad_copy()

        def function(self, m):
            from pyadjoint.reduced_functional_numpy import set_local
            set_local(self.tmpvec, m)
            ret = self.eps - self.rfn(self.tmpvec)
            return [ret]

        def jacobian(self, m):
            ret = self.rfn.derivative()
            ret._ad_mul(-1.)
            return [ret]

        def output_workspace(self):
            return [0.0]

        def length(self):
            """Return the number of components in the constraint vector (here, one)."""
            return 1
