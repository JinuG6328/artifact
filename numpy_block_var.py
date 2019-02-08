
from pyadjoint.tape import get_working_tape, annotate_tape, stop_annotating, no_annotations
from pyadjoint.block import Block
from pyadjoint.overloaded_type import (OverloadedType, FloatingType,
                                       create_overloaded_object, register_overloaded_type,
                                       get_overloaded_class)
import numpy as np

@register_overloaded_type
class array(FloatingType, np.array):
    def __init__(self, *args, **kwargs):
        super(array, self).__init__(*args,
                                    block_class=kwargs.pop("block_class", None),
                                    _ad_floating_active=kwargs.pop("_ad_floating_active", False),
                                    _ad_args=kwargs.pop("_ad_args", None),
                                    output_block_class=kwargs.pop("output_block_class", None),
                                    _ad_output_args=kwargs.pop("_ad_output_args", None),
                                    _ad_outputs=kwargs.pop("_ad_outputs", None),
                                    annotate=kwargs.pop("annotate", True),
                                    **kwargs)
        np.array.__init__(self, *args, **kwargs)

    def assign(self, *args, **kwargs):
        annotate_tape = kwargs.pop("annotate_tape", True)
        if annotate_tape:
            other = args[0]
            if not isinstance(other, OverloadedType):
                other = create_overloaded_object(other)

            block = AssignBlock(self, other)
            tape = get_working_tape()
            tape.add_block(block)

        # ret = backend.Constant.assign(self, *args, **kwargs)
        ret = np.array.__init__(self, *args, **kwargs)

        if annotate_tape:
            block.add_output(self.create_block_variable())

        return ret

    @classmethod
    def _ad_init_object(cls, obj):
        # r = np.zeros(obj.shape)

        r = obj.copy()
        return r
        
        # r = cls(obj.function_space())
        # r.vector()[:] = obj.vector()
        # return r

    @no_annotations
    def _ad_convert_type(self, value, options=None):
        pass

    def _ad_create_checkpoint(self):
        if self.block is None:
            # TODO: This might crash if annotate=False, but still using a sub-function.
            #       Because subfunction.copy(deepcopy=True) raises the can't access vector error.
            
            # TODO: what is the numpy copy capabilities we need? 
            # copy and copyto
            return self.copy()


        dep = self.block.get_dependencies()[0]
        
        # return backend.Function.sub(dep.saved_output, self.block.idx, deepcopy=True)
        # sub(self: dolfin.cpp.function.Function, arg0: int) → dolfin.cpp.function.Function
        # https://fenicsproject.org/docs/dolfin/2018.1.0/python/_autogenerated/dolfin.cpp.function.html?highlight=function%20sub#dolfin.cpp.function.Function.sub
        return dep.saved_output.copy()


    def _ad_restore_at_checkpoint(self, checkpoint):
        return checkpoint

    @no_annotations
    def adj_update_value(self, value):
        self.original_block_variable.checkpoint = value._ad_create_checkpoint()

    ## Question
    ## numpy array scalar or vector
    @no_annotations
    def _ad_mul(self, other):
        r = get_overloaded_class(np.array)
        np.copyto(self*other, r)
        return r

    @no_annotations
    def _ad_add(self, other):
        r = get_overloaded_class(np.array)
        np.copyto(self+other, r)
        return r

    def _ad_dot(self, other, options=None):
        
        # options = {} if options is None else options
        # riesz_representation = options.get("riesz_representation", "l2")
        
        return self.dot(other)

        # if riesz_representation == "l2":
        #     return self.vector().inner(other.vector())
        # elif riesz_representation == "L2":
        #     return backend.assemble(backend.inner(self, other) * backend.dx)
        # elif riesz_representation == "H1":
        #     return backend.assemble((backend.inner(self, other) + backend.inner(backend.grad(self), backend.grad(other))) * backend.dx)
        # else:
        #     raise NotImplementedError("Unknown Riesz representation %s" % riesz_representation)

    @staticmethod
    def _ad_assign_numpy(dst, src, offset):
        range_begin, range_end = (0,len(dst))
        # range_begin, range_end = dst.vector().local_range()
        dst[offset + range_begin:offset + range_end] = src[offset + range_begin:offset + range_end]
        
        # dst.vector().set_local(m_a_local)
        # dst.vector().apply('insert')
        offset += len(dst)
        return dst, offset

    @staticmethod
    def _ad_to_list(m):
        # if not hasattr(m, "gather"):
        #     m_v = m.vector()
        # else:
        #     m_v = m
        # m_a = gather(m_v)

        return m

    def _ad_copy(self):
        r = get_overloaded_class(np.array)
        r = self.copy()
        return r

    # def _ad_dim(self):
    #     return self.function_space().dim()

    def _ad_imul(self, other):
        # vec = self.vector()
        # scalar or vector
        vec *= other

    def _ad_iadd(self, other):
        # vec = self.vector()
        # FIXME: PETSc complains when we add the same vector to itself.
        # So we make a copy.
        vec += other#.vector().copy()

    def _reduce(self, r, r0):
        ## What is this?
        vec = self.vector().get_local()
        for i in range(len(vec)):
            r0 = r(vec[i], r0)
        return r0

    def _applyUnary(self, f):

        ## Similar to 
        ## ad_assign_numpy
        self = f
        
        # vec = self.vector()
        # npdata = vec.get_local()
        # for i in range(len(npdata)):
        #     npdata[i] = f(npdata[i])
        # vec.set_local(npdata)
        # vec.apply("insert")

    def _applyBinary(self, f, y):
        
        ## I think we don't need it.
        pass
        # vec = self.vector()
        # npdata = vec.get_local()
        # npdatay = y.vector().get_local()
        # for i in range(len(npdata)):
        #     npdata[i] = f(npdata[i], npdatay[i])
        # vec.set_local(npdata)
        # vec.apply("insert")

    def __deepcopy__(self, memodict={}):
        return self.copy()

## Lincom??
def _extract_functions_from_lincom(lincom, functions=None):
    functions = functions or []
    if isinstance(lincom, backend.Function):
        functions.append(lincom)
        return functions
    else:
        for op in lincom.ufl_operands:
            functions = _extract_functions_from_lincom(op, functions)
    return functions


# TODO: do we need an assign block?
class AssignBlock(Block):
    def __init__(self, func, other):
        super(AssignBlock, self).__init__()
        self.other = None
        self.lincom = False
        if isinstance(other, OverloadedType):
            self.add_dependency(other.block_variable, no_duplicates=True)
        else:
            # Assume that this is a linear combination
            functions = _extract_functions_from_lincom(other)
            for f in functions:
                self.add_dependency(f.block_variable, no_duplicates=True)
            self.expr = other
            self.lincom = True

    # def prepare_evaluate_adj(self, inputs, adj_inputs, relevant_dependencies):
    #     V = self.get_outputs()[0].output.function_space()
    #     adj_input_func = compat.function_from_vector(V, adj_inputs[0])

    #     if not self.lincom:
    #         return adj_input_func
    #     # If what was assigned was not a lincom (only currently relevant in firedrake),
    #     # then we need to replace the coefficients in self.expr with new values.
    #     replace_map = {}
    #     for dep in self.get_dependencies():
    #         replace_map[dep.output] = dep.saved_output
    #     expr = ufl.replace(self.expr, replace_map)
    #     return expr, adj_input_func

    def evaluate_adj_component(self, inputs, adj_inputs, block_variable, idx, prepared=None):
        if not self.lincom:
            if isinstance(block_variable.output, (AdjFloat, backend.Constant)):
                return adj_inputs[0].sum()
            else:
                adj_output = backend.Function(block_variable.output.function_space())
                adj_output.assign(prepared)
                return adj_output.vector()
        else:
            # # Linear combination
            # expr, adj_input_func = prepared
            # adj_output = backend.Function(block_variable.output.function_space())
            # diff_expr = ufl.algorithms.expand_derivatives(
            #     ufl.derivative(expr, block_variable.saved_output, adj_input_func)
            # )
            # adj_output.assign(diff_expr)
            return adj_inputs[0]

    # def prepare_evaluate_tlm(self, inputs, tlm_inputs, relevant_outputs):
    #     if not self.lincom:
    #         return None

    #     replace_map = {}
    #     for dep in self.get_dependencies():
    #         V = dep.output.function_space()
    #         tlm_input = dep.tlm_value or backend.Function(V)
    #         replace_map[dep.output] = tlm_input
    #     expr = ufl.replace(self.expr, replace_map)

    #     return expr

    def evaluate_tlm_component(self, inputs, tlm_inputs, block_variable, idx, prepared=None):
        # if not self.lincom:
        #     return tlm_inputs[0]

        # expr = prepared
        # V = block_variable.output.function_space()
        # tlm_output = backend.Function(V)
        # backend.Function.assign(tlm_output, expr)
        return tlm_inputs[0]

    # def prepare_evaluate_hessian(self, inputs, hessian_inputs, adj_inputs, relevant_dependencies):
    #     return self.prepare_evaluate_adj(inputs, hessian_inputs, relevant_dependencies)

    def evaluate_hessian_component(self, inputs, hessian_inputs, adj_inputs, block_variable, idx,
                                   relevant_dependencies, prepared=None):
        # Current implementation assumes lincom in hessian,
        # otherwise we need second-order derivatives here.
        # return self.evaluate_adj_component(inputs, hessian_inputs, block_variable, idx, prepared)
        return hessian_inputs[0]

    # def prepare_recompute_component(self, inputs, relevant_outputs):
    #     if not self.lincom:
    #         return None

    #     replace_map = {}
    #     for dep in self.get_dependencies():
    #         replace_map[dep.output] = dep.saved_output
    #     return ufl.replace(self.expr, replace_map)

    def recompute_component(self, inputs, block_variable, idx, prepared):
        
        #return Constant._constant_from_values(block_variable.output, inputs[0])

        return self
        # if not self.lincom:
        #     prepared = inputs[0]
        # output = backend.Function(block_variable.output.function_space())
        # backend.Function.assign(output, prepared)
        # return output
