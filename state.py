
from fenics import *
from fenics_adjoint import *

from initialize import *
from discretization import Discretization

class State(object):

    # def forward(self, ka):
    #     self.u, self.p = TrialFunctions(self.W)
    #     self.v, self.q = TestFunctions(self.W)
    #     self.w = get_state_variable(self.W)
    #     self.a = (inner( ka * self.u, self.v) + (div(self.v)*self.p) + (div(self.u)*self.q))*dx 
    #     self.n1 = FacetNormal(self.disc.mesh)
    #     self.myds = Measure('ds', domain=self.disc.mesh, subdomain_data=self.boundaries)
    #     self.L1 = dot(self.v,self.n1)*Constant(-1.)*self.myds(1)
    #     self.L2 = dot(self.v,self.n1)*Constant(1.)*self.myds(2)
    #     self.Amat, self.b = assemble_system(self.a, self.L1 + self.L2, self.bcs)
    #     solve(self.Amat, self.w.vector(), self.b)
    #     return self.w

    # def forward(self):
    #     self.u, self.p = TrialFunctions(self.W)
    #     self.v, self.q = TestFunctions(self.W)
    #     w = Function(self.W)
    #     self.a = (inner( self.ka * self.u, self.v) + (div(self.v)*self.p) + (div(self.u)*self.q))*dx 
    #     self.n1 = FacetNormal(self.disc.mesh)
    #     self.myds = Measure('ds', domain=self.disc.mesh, subdomain_data=self.boundaries)
    #     self.L1 = dot(self.v,self.n1)*Constant(-1.)*self.myds(1)
    #     self.L2 = dot(self.v,self.n1)*Constant(1.)*self.myds(2)
    #     self.Amat, self.b = assemble_system(self.a, self.L1 + self.L2, self.bcs)
    #     solve(self.Amat, w.vector(), self.b)
    #     return w

    # def add_args(parser):
    #     ''' Add options related to the mesh and discretization to the argument parser'''
    #     parser.add_argument("-m", "--mode", type=int, default=32, help="number of cells per grid direction")

    def __new__(self, disc, mode):
        self.disc = None
        self.W = None
        self.bcs = None
        self.w = None
        self.A = None
        self.V = None
        self.u, self.p = None, None
        self.v, self.q = None, None
        self.a = None
        self.n1 = None
        self.myds = None
        self.L1 = None
        self.L2 = None
        self.A, self.b = None, None
        self.k = None
        self.misfit_mode = None
        State.__init__(self, disc, mode)
        return self
        
    def __init__(self, disc, mode):
        self.disc = disc
        self.misfit_mode = mode

        import pdb
        pdb.set_trace()

        self.W, self.bcs = get_state_space(disc.mesh, disc.boundaries)
        self.w = get_state_variable(self.W)

        if self.misfit_mode:
            self.A = get_function_space(disc.mesh)
            self.V = Constant(0.5)
            self.k = interpolate(self.V, self.A)
        
        else:
            self.K = get_coefficient_space(disc.mesh)
            self.k = get_initial_coefficients(self.K)

        self.u, self.p = TrialFunctions(self.W)
        self.v, self.q = TestFunctions(self.W)
        self.a = (inner( self.k * self.u, self.v) + (div(self.v)*self.p) + (div(self.u)*self.q))*dx 
        self.n1 = FacetNormal(disc.mesh)
        self.myds = Measure('ds', domain=disc.mesh, subdomain_data=disc.boundaries)
        self.L1 = dot(self.v,self.n1)*Constant(-1.)*self.myds(1)
        self.L2 = dot(self.v,self.n1)*Constant(1.)*self.myds(2)
        self.Amat, self.b = assemble_system(self.a, self.L1 + self.L2, self.bcs)
        solve(self.Amat, self.w.vector(), self.b)
        
        # ## part for observation
        # self.w1 = get_state_variable(self.W)
        # self.K = get_coefficient_space(self.mesh)    
        # self.k =  get_initial_coefficients(self.K)


