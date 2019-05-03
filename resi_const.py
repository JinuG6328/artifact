from fenics import *
from fenics_adjoint import *
from numpy_block_var import Ndarray
import numpy as np

class ResidualConstraint(InequalityConstraint):
        """A class that enforces the volume constraint g(a) = V - a*dx >= 0."""

        def __init__(self, V, rfn, U = None):
            self.V = float(V)
            self.rfn = rfn
            self.U = U

        def function(self, m):
            if self.U is not None:
                # import pdb
                # pdb.set_trace()
                
                if isinstance(self.rfn.derivative(), Ndarray):
                    return [self.V - self.rfn(m)]
                else:
                    A = self.rfn.controls[0].function_space()
                    a = Function(A)
                    a.vector()[:] = self.U.dot(m)
                    return [self.V - self.rfn(a)]
            else:
                A = self.rfn.controls[0].function_space()
                a = Function(A)
                a.vector()[:] = m
                return [self.V - self.rfn(a)]

        def jacobian(self, m):
            if self.U is not None:
                if isinstance(self.rfn.derivative(), Ndarray):
                    return [-self.rfn.derivative()]
                else:
                    interm = self.U.T.dot(self.rfn.derivative().vector()[:])
                    return [-interm]
            else:
                return [-self.rfn.derivative().vector()[:]]

        def output_workspace(self):
            return [0.0]

        def length(self):
            """Return the number of components in the constraint vector (here, one)."""
            return 1