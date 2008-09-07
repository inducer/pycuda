from __future__ import division

__copyright__ = "Copyright (C) 2008 Andreas Kloeckner"

__license__ = """
Permission is hereby granted, free of charge, to any person
obtaining a copy of this software and associated documentation
files (the "Software"), to deal in the Software without
restriction, including without limitation the rights to use,
copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the
Software is furnished to do so, subject to the following
conditions:

The above copyright notice and this permission notice shall be
included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
OTHER DEALINGS IN THE SOFTWARE.
"""

import numpy
import pycuda.driver as drv
import pycuda.gpuarray as gpuarray
import pymbolic.mapper.substitutor




def dtype_to_ctype(dtype):
    dtype = numpy.dtype(dtype)
    if dtype == numpy.int32:
        return "int"
    elif dtype == numpy.uint32:
        return "unsigned int"
    elif dtype == numpy.int16:
        return "short int"
    elif dtype == numpy.uint16:
        return "short unsigned int"
    elif dtype == numpy.int8:
        return "signed char"
    elif dtype == numpy.uint8:
        return "unsigned char"
    elif dtype == numpy.intp or dtype == numpy.uintp:
        return "void *"
    elif dtype == numpy.float32:
        return "float"
    elif dtype == numpy.float64:
        return "double"
    else:
        raise ValueError, "unable to map dtype '%s'" % dtype




class DefaultingSubstitutionMapper(
        pymbolic.mapper.substitutor.SubstitutionMapper):
    def handle_unsupported_expression(self, expr):
        result = self.subst_func(expr)
        if result is not None:
            return result
        else:
            return expr

class CompiledVectorExpression(object):
    def __init__(self, vec_expr, type_getter, result_dtype, 
            stream=None, allocator=drv.mem_alloc):
        """
        @arg type_getter: A function `expr -> (is_vector, dtype)`, where
          C{is_vector} is a C{bool} determining whether C{expr} evaluates to 
          an aggregate over whose entries the kernel should iterate, and 
          C{dtype} is the numpy dtype of the expression.
        """
        self.result_dtype = result_dtype
        self.stream = stream
        self.allocator = allocator

        from pymbolic import get_dependencies
        deps = get_dependencies(vec_expr, composite_leaves=True)
        self.vector_exprs = [dep for dep in deps if type_getter(dep)[0]]
        self.scalar_exprs = [dep for dep in deps if not type_getter(dep)[0]]
        vector_names = ["v%d" % i for i in range(len(self.vector_exprs))]
        scalar_names = ["s%d" % i for i in range(len(self.scalar_exprs))]

        from pymbolic import substitute, var
        var_i = var("i")
        subst_map = dict(
                list(zip(self.vector_exprs, [var(vecname)[var_i]
                    for vecname in vector_names]))
                +list(zip(self.scalar_exprs, scalar_names)))
        def subst_func(expr):
            try:
                return subst_map[expr]
            except KeyError:
                return None

        subst_expr = DefaultingSubstitutionMapper(subst_func)(vec_expr)

        from pymbolic.mapper.stringifier import PREC_NONE, PREC_SUM
        from pymbolic.compiler import CompileMapper

        def get_c_declarator(name, is_vector, dtype):
            if is_vector:
                return "%s *%s" % (dtype_to_ctype(dtype), name)
            else:
                return "%s %s" % (dtype_to_ctype(dtype), name)
            
        from _kernel import get_scalar_kernel
        self.kernel = get_scalar_kernel(
                ", ".join([get_c_declarator("result", True, result_dtype)]+
                    [get_c_declarator(var_name, *type_getter(var_expr)) 
                        for var_expr, var_name in zip(
                            self.vector_exprs+self.scalar_exprs, 
                            vector_names+scalar_names)]),
                "result[i] = " + CompileMapper()(subst_expr, PREC_NONE),
                name="vector_expression"
                )

    def __call__(self, evaluate_subexpr):
        vectors = [evaluate_subexpr(vec_expr) for vec_expr in self.vector_exprs]
        scalars = [evaluate_subexpr(scal_expr) for scal_expr in self.scalar_exprs]

        from pytools import single_valued
        shape = single_valued(vec.shape for vec in vectors)
        result = gpuarray.empty(shape, self.result_dtype, self.stream, self.allocator)
        size = result.size
        
        self.kernel.prepared_async_call(vectors[0]._grid, self.stream,
                *([result.gpudata]+[v.gpudata for v in vectors]+scalars+[numpy.int32(size)]))

        return result




if __name__ == "__main__":
    import pycuda.autoinit
    from pymbolic import parse
    expr = parse("2*x+3*y+4*z")
    print expr
    cexpr = CompiledVectorExpression(expr, 
            lambda expr: (True, numpy.float32),
            numpy.float32)
    from pymbolic import var
    print cexpr({
        var("x"): gpuarray.arange(5),
        var("y"): gpuarray.arange(5),
        var("z"): gpuarray.arange(5),
        })

