from fenics import *
from fenics_adjoint import *

from initialize import *
from discretization import Discretization
from discretization_enc import Discretization_Enc
from state import *
from observation import *
from SVD import *

class Misfit_Enc(object):

    def __init__(self, namespace, disc, name = None, n_components):

        self.hello = None
        self.n_components = n_components
        
        self.power = 1.
        self.Alpha = 0.0
        
        self.disc1 = Discretization_Enc(self.n_components)
        
        self.ka = None

        self.obs = Observation(self.disc1)
        self.state = State(self.disc1)

        # Observation
        self.d_w = Function(self.state.W)
        if self.obs.observed == None:
            pass
        else:
            input_file = HDF5File(disc.mesh.mpi_comm(), "w.h5", "r")
            input_file.read(self.d_w, "Mixed")
            input_file.close()

        print("Misfit works")
        # self.n_components = 3
        # self.n_iter = 3
        # self.U, self.Sigma, self.VT = randomized_svd(self.Jhat, n_components= self.n_components, n_iter= self.n_iter, size = (disc.n+1)*(disc.n+1))

    def eval_cb(self, j, ka):
        self.ka_viz.assign(ka)
        self.controls << self.ka_viz
    
    def make_misfit_enc(self, d_w, Upa):
        self.controls = File("output/control_iterations_guess_Alpha(%f)_p(%f).pvd" % (self.Alpha, self.power) )
        self.ka_viz = Function(self.state.A, name="ControlVisualisation")

        self.ka = interpolate(self.state.V, self.state.A)

        self.w = self.state.solve(ka=self.ka)
        # for i in range(1,3):
        #     for j in range(1,3):
        #         self.e = Expression("sin(i*pi * x[0]) * sin(j*pi * x[1])", degree = 1, i = i, j = j)
        # self.f = interpolate(self.e,self.state.W.sub(1).collapse())
        # self.J = assemble((0.5*inner(self.w[1]-d_w[1], self.f))*dx)
        # self.J = self.J*self.J

        #self.J = assemble((0.5*inner(self.w[1]-d_w[1], self.w[1]-d_w[1])+0.5*inner(d_u-u, d_u-u))*dx + Alpha*(np.power(inner(grad(ka),grad(ka))+0.001,power))*dx)
        self.J = assemble(0.5*inner(self.w[1]-d_w[1], self.w[1]-d_w[1])*dx )
        
        self.m = Control(self.ka)
        # import pdb
        # pdb.set_trace()
        return self.J, self.ka

    def misfit(self, J, m):
        self.Jhat = ReducedFunctional(J, m, eval_cb_post=self.eval_cb)
        self.hello = compute_gradient(self.Jhat.functional, self.Jhat.controls[0])
        return self.Jhat

    
