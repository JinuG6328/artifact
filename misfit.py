
from fenics import *
from fenics_adjoint import *

from initialize import *
from discretization import Discretization
from state import *
from observation import *
from SVD import *

class Misfit(object):

    def __new__(self, namespace, disc):
        
        self.state = None
        self.obs = None
        
        self.Alpha = None
        self.d_w = None
        self.d_u = None
        self.d_p = None
        self.controls = None
        self.k_viz = None
        self.e = None
        self.f = None
        self.J = None
        self.m = None
        self.Jhat = None
        
        self.n_components = None
        self.n_iter = None
        self.power = None
        
        self.U, self.Sigma, self.VT = None, None, None
        Misfit.__init__(self, namespace, disc)
        return self

    def eval_cb(self, j, k):
        self.k_viz.assign(self.state.k)
        self.controls << self.k_viz
        
    def __init__(self, namespace, disc):

        self.n_components = 3
        self.n_iter = 3
        self.power = 1.

        self.obs = Observation(disc)
        self.state = State(disc,mode=True)
        
        # import pdb
        # pdb.set_trace()
        
        self.Alpha = Constant(0.0)
        self.d_w = Function(self.state.W)
        self.d_u,self.d_p = self.d_w.split(deepcopy=True)
        self.power = 1.

        input_file = HDF5File(disc.mesh.mpi_comm(), "w.h5", "r")
        input_file.read(self.d_w, "Mixed")
        input_file.close()

        # import pdb
        # pdb.set_trace()
        # input_file_p = HDF5File(self.disc.mesh.mpi_comm(), "p.h5", "r")
        # input_file_p.read(self.d_p, "Pressure")
        # input_file_p.close()

        # input_file_u = HDF5File(self.disc.mesh.mpi_comm(), "u.h5", "r")
        # input_file_u.read(self.d_u, "Velocity")
        # input_file_u.close()

        self.controls = File("output/control_iterations_guess_Alpha(%f)_p(%f).pvd" % (self.Alpha, self.power) )
        # import pdb
        # pdb.set_trace()
        self.k_viz = Function(self.state.A, name="ControlVisualisation")

        self.e = Expression("sin(pi * x[0]) * sin(pi * x[1])", degree = 1)
        self.f = interpolate(self.e,self.state.W.sub(1).collapse())
        self.J = assemble((0.5*inner(self.state.w[1]-self.d_w[1], self.f))*dx)
        #self.J = assemble((0.5*inner(self.state.w[1]-self.d_p, self.f))*dx)
        self.J = self.J*self.J

        
        self.m = Control(self.state.k)
        self.Jhat = ReducedFunctional(self.J, self.m, eval_cb_post=self.eval_cb)
        self.hello = compute_gradient(self.Jhat.functional, self.Jhat.controls[0])

        print("Misfit works")
        self.n_components = 3
        self.n_iter = 3
        self.U, self.Sigma, self.VT = randomized_svd(self.Jhat, n_components= self.n_components, n_iter= self.n_iter, size = (disc.n+1)*(disc.n+1))


