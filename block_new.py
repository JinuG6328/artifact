from fenics import *
from fenics_adjoint import *

from pyadjoint import Block
from pyadjoint.overloaded_type import OverloadedType

from dot_to_function import dot_to_function

backend_dot_to_function = dot_to_function

class UpdatedBlock(Block):
    def __init__(self, func, mat, array, **kwargs):
        super(UpdatedBlock, self).__init__()
        self.kwargs = kwargs
        self.umat = mat
        self.y = self.umat.T
        self.func = func
        self.add_dependency(array.block_variable)

    def __str__(self):
        return "UpdatedBlock"

    def recompute_component(self, inputs, block_variable, idx, prepared):
        return backend_dot_to_function(self.func,self.umat,inputs[0])

    def evaluate_adj_component(self, inputs, adj_inputs, block_variable, idx, prepared=None):
        adj_input = adj_inputs[0]
        return self.y.dot(adj_input)

    ## Doesn't called
    def evaluate_hessian_component(self, inputs, hessian_inputs, adj_inputs, block_variable, idx,
                                   relevant_dependencies, prepared=None):
        import pdb
        pdb.set_trace()
        return self.evaluate_adj_component(self, inputs, hessian_inputs, block_variable, idx)