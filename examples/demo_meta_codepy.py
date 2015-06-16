from __future__ import absolute_import
import pycuda.driver as cuda
import pycuda.autoinit
import numpy
import numpy.linalg as la
from pycuda.compiler import SourceModule
from six.moves import range

thread_strides = 16
block_size = 256
macroblock_count = 33

total_size = thread_strides*block_size*macroblock_count
dtype = numpy.float32

a = numpy.random.randn(total_size).astype(dtype)
b = numpy.random.randn(total_size).astype(dtype)

a_gpu = cuda.to_device(a)
b_gpu = cuda.to_device(b)
c_gpu = cuda.mem_alloc(a.nbytes)

from cgen import FunctionBody, \
        FunctionDeclaration, POD, Value, \
        Pointer, Module, Block, Initializer, Assign
from cgen.cuda import CudaGlobal

mod = Module([
    FunctionBody(
        CudaGlobal(FunctionDeclaration(
            Value("void", "add"),
            arg_decls=[Pointer(POD(dtype, name)) 
                for name in ["tgt", "op1", "op2"]])),
        Block([
            Initializer(
                POD(numpy.int32, "idx"),
                "threadIdx.x + %d*blockIdx.x" 
                % (block_size*thread_strides)),
            ]+[
            Assign(
                "tgt[idx+%d]" % (o*block_size),
                "op1[idx+%d] + op2[idx+%d]" % (
                    o*block_size, 
                    o*block_size))
            for o in range(thread_strides)]))])

mod = SourceModule(mod)

func = mod.get_function("add")
func(c_gpu, a_gpu, b_gpu, 
        block=(block_size,1,1),
        grid=(macroblock_count,1))

c = cuda.from_device_like(c_gpu, a)

assert la.norm(c-(a+b)) == 0

