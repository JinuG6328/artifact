from fenics import *
from fenics_adjoint import *

from pyadjoint import Block

from dot_to_function import dot_to_function

backend_dot_to_function = dot_to_function

class UpdatedBlock(Block):
	def __init__(self, func, mat, array, **kwargs):
		super(UpdatedBlock, self).__init__()
		self.kwargs = kwargs
		self.umat = mat
		self.ar = array
		self.y = self.umat.T
		# self.add_dependency(func.block_variable)

	def __str__(self):
		return "UpdatedBlock"

	def recompute_component(self, inputs, block_variable, idx, prepared):
		return backend_dot_to_function(inputs[0],inputs[1],inputs[2])

	def evaluate_adj_component(self, inputs, adj_inputs, block_variable, idx, prepared=None):
		adj_input = adj_inputs[0]
		return self.y.dot(adj_input)

	def evaluate_hessian_component(self, inputs, hessian_inputs, adj_inputs, block_variable, idx,
                                   relevant_dependencies, prepared=None):
		hessian_input = hessian_inputs[0]
		adj_input = adj_inputs[0]
		#y = self.umat.T
		y1 = self.y.T.dot(adj_input)
		y2 = hessian_input * np.matmul(self.y,self.y.T).dot(self.ar)
				
		return  y1 + y2
		
		"""This method must be overridden.

		The method should implement a routine for evaluating the hessian of the block.
		It is preferable that a "Forward-over-Reverse" scheme is used. Thus the hessians
		are evaluated in reverse (starting with the last block on the tape).

		"""
		# raise NotImplementedError(type(self))

		# def recompute_component(self, inputs, block_variable, idx, prepared):
		# 	return backend_normalise(inputs[0])