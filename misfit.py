from fenics import *
from fenics_adjoint import *

from initialize import *
from discretization import Discretization
from state import *
from observation import *
from SVD import *

class Misfit(object):

    def eval_cb(self, j, ka):
        self.ka_viz.assign(ka)
        self.controls << self.ka_viz
    
    def make_misfit(self, d_w):
        self.controls = File("output/control_iterations_guess_Alpha(%f)_p(%f).pvd" % (self.Alpha, self.power) )
        self.ka_viz = Function(self.state.A, name="ControlVisualisation")
        self.e = Expression("sin(pi * x[0]) * sin(pi * x[1])", degree = 1)
        self.w = self.state.solve()
        self.f = interpolate(self.e,self.state.W.sub(1).collapse())
        self.J = assemble((0.5*inner(self.w[1]-d_w[1], self.f))*dx)
        self.J = self.J*self.J
        self.m = Control(self.state.ka)
        return self.J, self.m

    def misfit(self, J, m):
        self.Jhat = ReducedFunctional(J, m, eval_cb_post=self.eval_cb)
        self.hello = compute_gradient(self.Jhat.functional, self.Jhat.controls[0])
        return self.Jhat

    def __init__(self, namespace, disc):

        self.hello = None
        self.n_components = 3
        self.n_iter = 3
        self.power = 1.
        self.Alpha = 0.0
        
        self.obs = Observation(disc)
        self.state = State(disc)

        # Observation
        self.d_w = Function(self.state.W)
        if self.obs.observed == None:
            pass
        else:
            input_file = HDF5File(disc.mesh.mpi_comm(), "w.h5", "r")
            input_file.read(self.d_w, "Mixed")
            input_file.close()

               
        # import pdb
        # pdb.set_trace()       

        print("Misfit works")
        # self.n_components = 3
        # self.n_iter = 3
        # self.U, self.Sigma, self.VT = randomized_svd(self.Jhat, n_components= self.n_components, n_iter= self.n_iter, size = (disc.n+1)*(disc.n+1))

