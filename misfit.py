
from fenics import *
from fenics_adjoint import *

from initialize import *
from discretization import Discretization
from state import *
from SVD import *

class Misfit(object):

    def __new__(self, state):
        self.Alpha = None
        self.state = state
        self.d_w = None
        self.controls = None
        self.ka_viz = None
        self.e = None
        self.f = None
        self.J = None
        self.m = None
        self.Jhat = None
        self.n_components = None
        self.n_iter = None
        self.power = None
        self.U, self.Sigma, self.VT = None, None, None
        Misfit.__init__(self)
        return self

    def eval_cb(self, j, ka):
        self.ka_viz.assign(self.state.ka)
        self.controls << self.ka_viz
        
    def __init__(self):

        self.n_components = 3
        self.n_iter = 3
        self.power = 1.

        self.disc = self.state.disc
        self.Alpha = Constant(0.0)
        self.d_w = Function(self.state.W)
        self.power = 1.

        input_file = HDF5File(self.disc.mesh.mpi_comm(), "w.h5", "r")
        input_file.read(self.d_w, "Mixed")
        input_file.close()

        self.controls = File("output/control_iterations_guess_Alpha(%f)_p(%f).pvd" % (self.Alpha, self.power) )
        self.ka_viz = Function(self.state.A, name="ControlVisualisation")

        self.e = Expression("sin(pi * x[0]) * sin(pi * x[1])", degree = 1)
        self.f = interpolate(self.e,self.state.W.sub(1).collapse())
        self.J = assemble((0.5*inner(self.state.w[1]-self.d_w[1], self.f))*dx)
        self.J = self.J*self.J

        self.m = Control(self.state.ka)
        self.Jhat = ReducedFunctional(self.J, self.m, eval_cb_post=self.eval_cb)
        self.hello = compute_gradient(self.Jhat.functional, self.Jhat.controls[0])

        self.n_components = 3
        self.n_iter = 3
        self.U, self.Sigma, self.VT = randomized_svd(self.Jhat, n_components= self.n_components, n_iter= self.n_iter, size = (self.disc.n+1)*(self.disc.n+1))


