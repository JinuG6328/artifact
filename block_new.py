from fenics import *
from fenics_adjoint import *
from pyadjoint import Block

class UpdatedBlock(Block):
	def __init__(self, func, mat, **kwargs):
		super(UpdatedBlock, self).__init__()
		self.kwargs = kwargs
		self.umat = mat
		self.add_dependency(func.block_variable)

	def __str__(self):
    	return "UpdatedBlock"

	def evaluate_adj_component(self, inputs, adj_inputs, block_variable, idx, prepared=None):
		adj_input = adj_inputs[0]
		# self.umat.shape[1]
		x = inputs[idx].vector()
		#inv_xnorm = 1.0 / x.norm('l2')
		y = self.umat.T.dot(x)
		return y.inner(adj_input)

	def evaluate_hessian_component(self, inputs, hessian_inputs, adj_inputs, block_variable, idx,
                                   relevant_dependencies, prepared=None):
        """This method must be overridden.

        The method should implement a routine for evaluating the hessian of the block.
        It is preferable that a "Forward-over-Reverse" scheme is used. Thus the hessians
        are evaluated in reverse (starting with the last block on the tape).

        """
        raise NotImplementedError(type(self))
	
	# def recompute_component(self, inputs, block_variable, idx, prepared):
    # 	return backend_normalise(inputs[0])