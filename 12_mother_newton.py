from dolfin import *
from dolfin_adjoint import *

import numpy as np
from scipy import linalg, sparse
from sklearn.utils import *
from sklearn.utils.extmath import svd_flip

def safe_sparse_dot(a, b):
    
    if isinstance(a, ReducedFunctional):
        # First: get the function space
        fs = rf.controls[0].coeff.function_space()
        q_dot = Function(fs)
        c_dot = Function(fs)
        c = np.ndarray(b.shape)
        print(c.shape)
        for i in range(len(b.T)):
            #import pdb
            #pdb.set_trace()
            print(b.T[i])
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
    Q = randomized_range_finder(rf, n_random, n_iter, size, power_iteration_normalizer, random_state)
                                #(A, size, n_iter, Size_f_rf, power_iteration_normalizer='auto', random_state=None):

    # Change m to rf
    # project M to the (k + p) dimensional space using the basis vectors
    #
    B = safe_sparse_dot(rf, Q)
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



# Declare mesh and function spaces
Size = 64
mesh = UnitSquareMesh(Size, Size)

V = FunctionSpace(mesh, "CG", 1) # state space
W = FunctionSpace(mesh, "DG", 0) # control space

z = interpolate(Constant(1), W) # zero initial guess
z.rename("Control", "Control")
u = Function(V, name='State')
v = TestFunction(V)

# Define and solve the Poisson equation
F = (inner(grad(u), grad(v)) - z*v)*dx
bc = DirichletBC(V, 0.0, "on_boundary")
solve(F == 0, u, bc)

# Turn off forward and adjoint solver output
set_log_level(ERROR)

# Define regularisation parameter
alphaval = 0.
alpha = Constant(alphaval)

# Define observation data
x = SpatialCoordinate(mesh)
d = (1/(2*pi**2) + 2*alpha*pi**2)*sin(pi*x[0])*sin(pi*x[1])

# Define functional of interest and the reduced functional
J = Functional((0.5*inner(u-d, u-d))*dx + 0.5*alpha*z**2*dx)

# Define Control and ReductionFuctional
control = Control(z)
rf = ReducedFunctional(J, control)
print(type(rf))
print(isinstance(rf,ReducedFunctional))

# Define z and u 
q = interpolate(Expression("2*pi*pi*sin(pi*x[0])*sin(pi*x[1])", degree = 2), W)

print(type(q.vector()))
vel = interpolate(Expression("sin(pi*x[0])*sin(pi*x[1])", degree = 2), W)

# Calculate reducedfunctional hessian
x = [0 for i in range(64)]
x[0] = 1
print(x)
H1q = rf.hessian(q)
#H2q = rf.hessian(x)
print(H1q)
print(type(H1q))

n_components = 10
n_iter = 5
U, Sigma, VT = randomized_svd(rf, n_components= n_components, n_iter= n_iter, size = 2*Size*Size) # size should be the discrete vector size of q
print(Sigma)

# Validate whether hessian works
K = q.vector().inner(H1q.vector())
print(('K=',K))

# print(len(H1q.vector().array()))
# print(H1q)
# print(len(q.vector().array()))
#H = hessian(J, control)
#K1 = H(q).vector().inner(q.vector())
#print(type(K))
#print(K1)

## todo
## check the u value.

z_squared=q.vector().inner(q.vector())
#u_squared = vel.vector().inner(vel.vector())
u_squared = vel.vector().inner(vel)
print(z_squared)
print(u_squared)
K2 = alphaval * z_squared - u_squared
print (K2)


# z <- solution value 2*pi**2 sin(pi x) sin (pi y)
# u <- solution value of u is sin(pi x) sin (pi y) ? verify
# z^T H z = alpha \| z \|^2 - \|u \|^2 ?



# Define optimisation problem
problem = MinimizationProblem(rf)

# Now configure the optimisation solver
params_dict = {
    'General': {
        'Krylov': {
            'Type': 'Conjugate Gradients',
            'Absolute Tolerance': 1.0e-5,
            'Relative Tolerance': 1.0e-5,
            'Iteration Limit': 50,
        },
    },
    'Step': {
        'Type': 'Line Search',
        'Line Search': {
            'Descent Method': {
                'Type': 'Newton-Krylov'
            },
        }
    },
    'Status Test': {
        'Gradient Tolerance': 1e-8,
    }
}
parameters = {"acceptable_tol": 1.0e-3, "maximum_iterations": 100}
#solver = ROLSolver(problem, parameters = params_dict)
solver = IPOPTSolver(problem, parameters = parameters)
z_opt = solver.solve()

File("output/12/control.pvd") << z_opt

