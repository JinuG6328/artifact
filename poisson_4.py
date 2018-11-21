from fenics import *
from fenics_adjoint import *
import sympy as sym
import moola
import numpy as np
import math

from scipy import linalg, sparse
from sklearn.utils import *
from sklearn.utils.extmath import svd_flip


# TODO: break out this into a separate file
def safe_sparse_dot(a, b):
    
    if isinstance(a, ReducedFunctional):
        # First: get the function space
        # import pdb;
        # pdb.set_trace()
        fs = a.controls[0].coeff.function_space()
        print("fs", type(fs), fs)
        q_dot = Function(fs)
        c_dot = Function(fs)
        c = np.ndarray(b.shape)
        print(c.shape)
        for i in range(len(b.T)):
            
            print("b", b.T[i])
            print("len", len(b.T[i]))
            print("q", q_dot.vector()[:])
            print("len_q", len(q_dot.vector()[:]))
            # import pdb
            # pdb.set_trace()
            q_dot.vector()[:] = numpy.ascontiguousarray(b.T[i])


            c_dot = a.hessian(q_dot,project=True)
            c[:,i] = c_dot.vector()[:]
            print(c[i])
        print(c)
        return c
    else:
        return np.dot(a, b)

def randomized_range_finder(A, size, n_iter, Size_f_rf, power_iteration_normalizer='auto', random_state=None):
    """Computes an orthonormal matrix whose range approximates the range of A.

    Parameters
    ----------
    A : 2D array
        The input data matrix

    size : integer
        Size of the return array

    n_iter : integer
        Number of power iterations used to stabilize the result

    power_iteration_normalizer : 'auto' (default), 'QR', 'LU', 'none'
        Whether the power iterations are normalized with step-by-step
        QR factorization (the slowest but most accurate), 'none'
        (the fastest but numerically unstable when `n_iter` is large, e.g.
        typically 5 or larger), or 'LU' factorization (numerically stable
        but can lose slightly in accuracy). The 'auto' mode applies no
        normalization if `n_iter` <= 2 and switches to LU otherwise.

        .. versionadded:: 0.18

    random_state : int, RandomState instance or None, optional (default=None)
        The seed of the pseudo random number generator to use when shuffling
        the data.  If int, random_state is the seed used by the random number
        generator; If RandomState instance, random_state is the random number
        generator; If None, the random number generator is the RandomState
        instance used by `np.random`.

    Returns
    -------
    Q : 2D array
        A (size x size) projection matrix, the range of which
        approximates well the range of the input matrix A.

    Notes
    -----

    Follows Algorithm 4.3 of
    Finding structure with randomness: Stochastic algorithms for constructing
    approximate matrix decompositions
    Halko, et al., 2009 (arXiv:909) http://arxiv.org/pdf/0909.4061

    An implementation of a randomized algorithm for principal component
    analysis
    A. Szlam et al. 2014
    """
    random_state = check_random_state(random_state)

    # Generating normal random vectors with shape: (A.shape[1], size)
    Q = random_state.normal(size=(Size_f_rf, size)).astype('float_')
    # if A.dtype.kind == 'f':
    #     # Ensure f32 is preserved as f32
    #     Q = Q.astype(A.dtype, copy=False)

    # Deal with "auto" mode
    if power_iteration_normalizer == 'auto':
        if n_iter <= 2:
            power_iteration_normalizer = 'none'
        else:
            power_iteration_normalizer = 'LU'

    # Perform power iterations with Q to further 'imprint' the top
    # singular vectors of A in Q
    for i in range(n_iter):
        if power_iteration_normalizer == 'none':
            Q = safe_sparse_dot(A, Q)
            Q = safe_sparse_dot(A.T, Q)
        elif power_iteration_normalizer == 'LU':
            Q, _ = linalg.lu(safe_sparse_dot(A, Q), permute_l=True)
            # TODO: rf.hessian transpose?
            Q, _ = linalg.lu(safe_sparse_dot(A, Q), permute_l=True)
        elif power_iteration_normalizer == 'QR':
            Q, _ = linalg.qr(safe_sparse_dot(A, Q), mode='economic')
            # TODO: rf.hessian transpose?
            Q, _ = linalg.qr(safe_sparse_dot(A, Q), mode='economic')

    # Sample the range of A using by linear projection of Q
    # Extract an orthonormal basis
    Q, _ = linalg.qr(safe_sparse_dot(A, Q), mode='economic')
    return Q



# Adding size component
# remove transpose
def randomized_svd(M, n_components, n_oversamples=10, n_iter='auto',
                   power_iteration_normalizer='auto',
                   flip_sign=True, random_state=0, size = 32):
    """Computes a truncated randomized SVD

    Parameters
    ----------
    M : ndarray or sparse matrix
        Matrix to decompose

    n_components : int
        Number of singular values and vectors to extract.

    n_oversamples : int (default is 10)
        Additional number of random vectors to sample the range of M so as
        to ensure proper conditioning. The total number of random vectors
        used to find the range of M is n_components + n_oversamples. Smaller
        number can improve speed but can negatively impact the quality of
        approximation of singular vectors and singular values.

    n_iter : int or 'auto' (default is 'auto')
        Number of power iterations. It can be used to deal with very noisy
        problems. When 'auto', it is set to 4, unless `n_components` is small
        (< .1 * min(X.shape)) `n_iter` in which case is set to 7.
        This improves precision with few components.

        .. versionchanged:: 0.18

    power_iteration_normalizer : 'auto' (default), 'QR', 'LU', 'none'
        Whether the power iterations are normalized with step-by-step
        QR factorization (the slowest but most accurate), 'none'
        (the fastest but numerically unstable when `n_iter` is large, e.g.
        typically 5 or larger), or 'LU' factorization (numerically stable
        but can lose slightly in accuracy). The 'auto' mode applies no
        normalization if `n_iter` <= 2 and switches to LU otherwise.

        .. versionadded:: 0.18

    transpose : True, False or 'auto' (default)
        Whether the algorithm should be applied to M.T instead of M. The
        result should approximately be the same. The 'auto' mode will
        trigger the transposition if M.shape[1] > M.shape[0] since this
        implementation of randomized SVD tend to be a little faster in that
        case.

        .. versionchanged:: 0.18

    flip_sign : boolean, (True by default)
        The output of a singular value decomposition is only unique up to a
        permutation of the signs of the singular vectors. If `flip_sign` is
        set to `True`, the sign ambiguity is resolved by making the largest
        loadings for each component in the left singular vectors positive.

    random_state : int, RandomState instance or None, optional (default=None)
        The seed of the pseudo random number generator to use when shuffling
        the data.  If int, random_state is the seed used by the random number
        generator; If RandomState instance, random_state is the random number
        generator; If None, the random number generator is the RandomState
        instance used by `np.random`.

    Notes
    -----
    This algorithm finds a (usually very good) approximate truncated
    singular value decomposition using randomization to speed up the
    computations. It is particularly fast on large matrices on which
    you wish to extract only a small number of components. In order to
    obtain further speed up, `n_iter` can be set <=2 (at the cost of
    loss of precision).

    References
    ----------
    * Finding structure with randomness: Stochastic algorithms for constructing
      approximate matrix decompositions
      Halko, et al., 2009 http://arxiv.org/abs/arXiv:0909.4061

    * A randomized algorithm for the decomposition of matrices
      Per-Gunnar Martinsson, Vladimir Rokhlin and Mark Tygert

    * An implementation of a randomized algorithm for principal component
      analysis
      A. Szlam et al. 2014
    """
    # if isinstance(M, (sparse.lil_matrix, sparse.dok_matrix)):
    #     warnings.warn("Calculating SVD of a {} is expensive. "
    #                   "csr_matrix is more efficient.".format(
    #                       type(M).__name__),
    #                   sparse.SparseEfficiencyWarning)

    random_state = check_random_state(random_state)
    M_shape=(size, size)
    n_random = n_components + n_oversamples
    n_samples, n_features = M_shape

    if n_iter == 'auto':
        # Checks if the number of iterations is explicitly specified
        # Adjust n_iter. 7 was found a good compromise for PCA. See #5299
        n_iter = 7 if n_components < .1 * min(M_shape) else 4

    # if transpose == 'auto':
    #     transpose = n_samples < n_features
    # if transpose:
    #     # this implementation is a bit faster with smaller shape[1]
    #     M = M.T

    # Change m to rf
    # import pdb
    # pdb.set_trace()
    Q = randomized_range_finder(M, n_random, n_iter, size, power_iteration_normalizer, random_state)
                              #(A, size, n_iter, Size_f_rf, power_iteration_normalizer='auto', random_state=None):

    # Change m to rf
    # project M to the (k + p) dimensional space using the basis vectors
    #
    B = safe_sparse_dot(M, Q)
    B = B.T

    # compute the SVD on the thin matrix: (k + p) wide
    Uhat, s, V = linalg.svd(B, full_matrices=False)

    del B
    U = np.dot(Q, Uhat)

    # change it to just U, V
    # if flip_sign:
    #     if not transpose:
    #         U, V = svd_flip(U, V)
    #     else:
    #         # In case of transpose u_based_decision=false
    #         # to actually flip based on u and not v.
    #         U, V = svd_flip(U, V, u_based_decision=False)
    U, V = svd_flip(U, V, u_based_decision=False)
    # if transpose:
    #     # transpose back the results according to the input convention
    #     return V[:n_components, :].T, s[:n_components], U[:, :n_components].T
    # else:
    return U[:, :n_components], s[:n_components], V[:n_components, :]

set_log_level(ERROR)

class Left(SubDomain):
	def inside(self, x, on_boundary):
		return near(x[0], 0.0)

class Right(SubDomain):
	def inside(self, x, on_boundary):
		return near(x[0], 1.0)

class Top(SubDomain):
	def inside(self, x, on_boundary):
		return near(x[1], 1.0)

class Bottom(SubDomain):
	def inside(self, x, on_boundary):
		return near(x[1], 0.0)

def get_mesh(N):

	#making mesh
	mesh = UnitSquareMesh(N,N)

	#making boundary
	left = Left()
	right = Right()
	top = Top()
	bottom = Bottom()
	boundaries = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
	boundaries.set_all(0)
	left.mark(boundaries, 1)
	right.mark(boundaries, 2)
	top.mark(boundaries, 3)
	bottom.mark(boundaries, 4)
	return mesh, boundaries

def get_state_space(mesh, boundaries=None):

	#setting two variable
	V = FiniteElement("RT", mesh.ufl_cell(), 2)
	Q = FiniteElement("DG", mesh.ufl_cell(), 1)

	#making mixed space
	VQ = V*Q
	W = FunctionSpace(mesh,VQ)

	if boundaries:
		
		bcu_01 = DirichletBC(W.sub(0), Constant((0.0,0.0)), boundaries, 3)
		bcu_02 = DirichletBC(W.sub(0), Constant((0.0,0.0)), boundaries, 4)
	
		bcs = [bcu_01, bcu_02]
	else:
		bcs = None

	return W, bcs

def get_state_variable(W):
	return Function(W)

def get_function_space(mesh):
	A = FunctionSpace(mesh, 'CG', 1)
	return A

def alpha(ka):
	return ka

def forward_problem(ka):
	(u,p) = TrialFunctions(W)
	(v,q) = TestFunctions(W)
	a = (inner( alpha(ka) * u,v) + (div(v)*p) + (div(u)*q))*dx 
	n = FacetNormal(mesh)
	myds = Measure('ds', domain=mesh, subdomain_data=boundaries)
	L1 = dot(v,n)*Constant(-1.)*myds(1)
	L2 = dot(v,n)*Constant(1.)*myds(2)
	l = L1 + L2 
	solve(a==l, w, bcs)
	return w

Size = 32
mesh, boundaries = get_mesh(Size)
W, bcs = get_state_space(mesh, boundaries)
w = get_state_variable(W)
A = get_function_space(mesh)
V = Constant(0.5)
Alpha = Constant(0.001)
power = 1.

d_p = Function(W.sub(1).collapse())
input_file = HDF5File(mesh.mpi_comm(), "p_n.h5", "r")
input_file.read(d_p, "Pressure")
input_file.close()

d_u = Function(W.sub(0).collapse())
input_file = HDF5File(mesh.mpi_comm(), "u_n.h5", "r")
input_file.read(d_u, "Velocity")
input_file.close()

print(type(d_u))

if __name__ == "__main__":
	
	ka = interpolate(V, A, name="Control") # initial guess.
	w = forward_problem(ka) 
	(u,p) = split(w)
	#(u,p) = w.split(True)

	controls = File("output/control_iterations_guess_Alpha(%f)_p(%f).pvd" % (Alpha, power) )
	ka_viz = Function(A, name="ControlVisualisation")
    
	def eval_cb(j, ka):
		ka_viz.assign(ka)
		controls << ka_viz
	
        # TODO: see if we can construct a J consisting of a pressure at fixed number of evaluation points
	J = Functional((0.5*inner(d_p-p, d_p-p)+0.5*inner(d_u-u, d_u-u))*dx + Alpha*(np.power(inner(grad(ka),grad(ka))+0.001,power))*dx)
	#J = Functional((0.5*inner(d_u-u, d_u-u))*dx + Alpha*(np.power(inner(grad(ka),grad(ka))+0.001,power))*dx)
	#J = Functional((0.5*inner(d_p-p, d_p-p))*dx + Alpha*(np.power(inner(grad(ka),grad(ka))+0.001,power))*dx)

	#norm
	m = Control(ka)
	Jhat = ReducedFunctional(J, m, eval_cb_post=eval_cb)
	# H = hessian(J, m)
	# print(type(H))
	n_components = 100
	n_iter = 5
        # TODO: use A -- the function space of the parameter -- to get the size
	U, Sigma, VT = randomized_svd(Jhat, n_components= n_components, n_iter= n_iter, size = (Size+1)*(Size+1)) # size should be the discrete vector size of q
    # This if for RT
	print(Sigma)
# 	lb = 0.0
# 	ub = 1.0
# #224 the paper
# 	problem = MinimizationProblem(Jhat, bounds=(lb, ub))

# 	parameters = {"acceptable_tol": 1.0e-3, "maximum_iterations": 100}
# 	solver = IPOPTSolver(problem, parameters=parameters)
# 	ka_opt = solver.solve()

# 	xdmf_filename = XDMFFile("output/final_solution_Alpha(%f)_p(%f).xdmf" % (Alpha ,power))
# 	xdmf_filename.write(ka_opt)

# 	(u1,p1) = w.split(True)
# 	velocity_opt = File("opt_velocity.pvd")
# 	V2 = W.sub(0).collapse()
# 	velo_viz = Function(V2, name="velocity")
# 	velo_viz.assign(u1)
# 	velocity_opt << velo_viz


# 	pressure_opt = File("opt_pressure.pvd")
# 	V3 = W.sub(1).collapse()
# 	pressure_viz = Function(V3, name="pressure")
# 	pressure_viz.assign(p1)
# 	pressure_opt << pressure_viz
