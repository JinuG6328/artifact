from fenics import *
from fenics_adjoint import *

from pyadjoint import Block
from pyadjoint.overloaded_function import overload_function

import numpy as np

def Log_(input):
    return np.log(input.real)

class LogBlock(Block):
    def __init__(self, func, **kwargs):
        super(LogBlock, self).__init__()
        self.kwargs = kwargs
        self.add_dependency(func)

    def __str__(self):
        return 'LogBlock'

    def evaluate_adj_component(self, inputs, adj_inputs, block_variable, idx, prepared=None):
        adj_input = adj_inputs[0]
        return adj_input / inputs[0]

    def recompute_component(self, inputs, block_variable, idx, prepared):
        import pdb
        pdb.set_trace()
        return np.log(inputs[0])


Log = overload_function(Log_, LogBlock)