from fenics import *
from fenics_adjoint import *
import numpy as np

class Projection(object):

    def __init__(self, U, ka_opt):
        projector = np.matmul(U.T,U)
        projector1 = np.linalg.pinv(projector)
        projector2 = np.matmul(U, projector1)
        projector3 = np.matmul(projector2, U.T)

        sample = ka_opt.copy(deepcopy=True)
        length_array = len(sample.vector()[:])
        sample.vector()[:] = np.zeros(length_array)
        sample.vector()[int(length_array/2)] = 1
        osc = np.matmul(projector3, sample.vector()[:])
