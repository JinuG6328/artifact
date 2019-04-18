from fenics import *
from fenics_adjoint import *

class ResidualConstraint(InequalityConstraint):
        """A class that enforces the volume constraint g(a) = V - a*dx >= 0."""

        def __init__(self, V, rfn, U):
            self.V = float(V)
            self.rfn = rfn
            self.U = U

            # self.smass = assemble(TestFunction(A) * Constant(1) * dx)
            # self.tmpvec = Function(A)

        def function(self, m):
            # from pyadjoint.reduced_functional_numpy import set_local
            # set_local(self.tmpvec, m)

            # # Compute the integral of the control over the domain

            # integral = self.smass.inner(self.tmpvec.vector())
            A = self.rfn.controls[0].function_space()
            a = Function(A)
            a.vector()[:] = self.U.dot(m)
            return [self.V - self.rfn(a)]
            # return [self.V - integral]

        def jacobian(self, m):
            # import pdb
            # pdb.set_trace()
            interm = self.U.T.dot(self.rfn.derivative().vector()[:])
            return [interm]

        def output_workspace(self):
            return [0.0]

        def length(self):
            """Return the number of components in the constraint vector (here, one)."""
            return 1