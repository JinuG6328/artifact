from fenics import *
from fenics_adjoint import *

from pyadjoint import Block

# from dot_to_function import dot_to_function

# backend_dot_to_function = dot_to_function

class UpdatedBlock_arr(Block):
	def __init__(self, array, **kwargs):
		super(UpdatedBlock_arr, self).__init__()
		self.kwargs = kwargs
		self.ar = array
		self.num = 1

	def __str__(self):
		return "UpdatedBlock_arr"

	def recompute_component(self, inputs, block_variable, idx, prepared):
		return self.ar

	def evaluate_adj_component(self, inputs, adj_inputs, block_variable, idx, prepared=None):
		return self.num

	def evaluate_hessian_component(self, inputs, hessian_inputs, adj_inputs, block_variable, idx,
                                   relevant_dependencies, prepared=None):			
		return  self.num
		
		"""This method must be overridden.

		The method should implement a routine for evaluating the hessian of the block.
		It is preferable that a "Forward-over-Reverse" scheme is used. Thus the hessians
		are evaluated in reverse (starting with the last block on the tape).

		"""
		# raise NotImplementedError(type(self))

		# def recompute_component(self, inputs, block_variable, idx, prepared):
		# 	return backend_normalise(inputs[0])