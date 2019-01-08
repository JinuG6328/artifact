
from fenics import *
from fenics_adjoint import *

from initialize import *
from discretization import Discretization
from forward import *

class Misfit(Forward):

    def __init__(self):
        Forward.__init__(self)
        self.d_w = None
        self.controls = None
        self.ka_viz = None
        self.e = None
        self.f = None
        self.J = None
        self.m = None
        self.Jhat = None
        self.n_components = 3
        self.n_iter = 3
        self.U, self.Sigma, self.VT = None, None, None
        
    def __new__(self, Forward):

        self.d_w = Function(self.W)
        input_file = HDF5File(mesh.mpi_comm(), "w.h5", "r")
        input_file.read(self.d_w, "Mixed")
        input_file.close()

        self.controls = File("output/control_iterations_guess_Alpha(%f)_p(%f).pvd" % (Alpha, power) )
        self.ka_viz = Function(self.A, name="ControlVisualisation")
    
        def eval_cb(j, ka):
            self.ka_viz.assign(self.ka)
            self.controls << self.ka_viz

        self.e = Expression("sin(pi * x[0]) * sin(pi * x[1])", degree = 1)
        self.f = interpolate(self.e,self.W.sub(1).collapse())
        self.J = assemble((0.5*inner(self.w[1]-self.d_w[1], self.f))*dx)
        self.J = self.J*self.J

        self.m = Conrol(self.ka)
        self.Jhat = ReducedFunctional(self.J, self.m, eval_cb_post=self.eval_cb)
        self.hello = compute_gradient(self.Jhat.functional, self.Jhat.controls[0])

        self.n_components = 3
        self.n_iter = 3
        self.U, self.Sigma, self.VT = randomized_svd(self.Jhat, n_components= self.n_components, n_iter= self.n_iter, size = (self.n+1)*(self.n+1))
        return self


