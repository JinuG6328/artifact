from fenics import *
from fenics_adjoint import *

from initialize import *
from discretization import Discretization
from state import *
from observation import *
from SVD import *


import matplotlib.pyplot as plt

class Misfit(object):

    def __init__(self, namespace, disc, name = None):

        self.hello = None
        self.n_components = 3
        self.n_iter = 3
        self.power = 1.
        self.Alpha = 0.0
        
        self.ka = None

        self.disc = disc
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

        print("Misfit works")

    def add_args(parser):
        parser.add_argument("-nc", "--num_component", type=int, default=9, help="add noise with specific standard deviation")

    # def eval_cb(self, j, ka):
    #     if isinstance(ka, Function):
    #         self.ka_viz.assign(ka)
    #         self.controls << self.ka_viz
    
    def prediction_center(self, ka, x_0 = 0.5, y_0 = 0.8):

        self.ka = ka
        self.w = self.state.solve(ka=self.ka)
        
        sigma = 0.1
        q_expr = Expression("exp(-(0.5/sigma)*((x[0]-x_0)*(x[0]-x_0)+(x[1]-y_0)*(x[1]-y_0)))", x_0 = x_0, y_0 = y_0, sigma = sigma, degree = 3)
        q_adjflt = assemble(q_expr*self.w[1]*dx)
    
        return q_adjflt

    def prediction_boundaries(self, ka):
        self.n1 = FacetNormal(self.disc.mesh)
        self.myds = Measure('ds', domain=self.disc.mesh, subdomain_data=self.disc.boundaries)

        self.w = self.state.solve(ka=self.ka)
        self.J = assemble(inner(self.w, self.w)*myds(1))
        pass

    def make_misfit_red(self, d_w, ka):
        ## TODO
        ## input for the number of components
        # self.controls = File("output/control_iterations_guess_Alpha(%f)_p(%f).pvd" % (self.Alpha, self.power) )
        # self.ka_viz = Function(self.state.A, name="ControlVisualisation")

        self.ka = ka

        self.w = self.state.solve(ka=self.ka)
        self.f = Function(self.state.A)
        self.J = assemble(inner(self.f,self.f)*dx)

        for i in range(1,4):
            for j in range(1,4):
                self.e = Expression("sin(i*pi * x[0]) * sin(j*pi * x[1])", degree = 9, i = i, j = j)
                self.mid = interpolate(self.e,self.state.W.sub(1).collapse())
                self.J_int = assemble((0.5*inner(self.w[1]-d_w[1], self.mid))*dx)
                self.J_int_2 = self.J_int*self.J_int
                self.J += self.J_int_2
        return self.J

    def make_misfit(self, d_w, ka):
        # self.controls = File("output/control_iterations_guess_Alpha(%f)_p(%f).pvd" % (self.Alpha, self.power) )
        # self.ka_viz = Function(self.state.A, name="ControlVisualisation")
        self.ka = ka

        self.w = self.state.solve(ka=self.ka)
        self.J = assemble(0.5*inner(self.w[1]-d_w[1], self.w[1]-d_w[1])*dx )
        return self.J
        
    def misfit_op(self, J, m):
        self.Jhat = ReducedFunctional(J, m)
        self.hello = compute_gradient(self.Jhat.functional, self.Jhat.controls[0])
        return self.Jhat

    
