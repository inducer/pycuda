import pycuda.driver as cuda
import pycuda.autoinit
import numpy
import numpy.linalg as la
from pycuda.compiler import SourceModule

block_size = 16
thread_block_size = 32
macroblock_count = 33
dtype = numpy.float32

a = numpy.random.randn(block_size*thread_block_size*macroblock_count)\
        .astype(dtype)
b = numpy.random.randn(block_size*thread_block_size*macroblock_count)\
        .astype(dtype)

a_gpu = cuda.to_device(a)
b_gpu = cuda.to_device(b)
c_gpu = cuda.mem_alloc(a.nbytes)

from codepy.cgen import FunctionBody, FunctionDeclaration, \
        Typedef, POD, Value, Pointer, Module, Block, Initializer, Assign

from codepy.cgen.cuda import CudaGlobal
mod = Module([
    Typedef(POD(dtype, "value_type")),
    FunctionBody(
        CudaGlobal(FunctionDeclaration(
            Value("void", "add"),
            [Pointer(POD(dtype, name)) for name in ["result", "op1", "op2"]])),
        Block([
            Initializer(
                POD(numpy.int32, "idx"),
                "threadIdx.x + %d*blockIdx.x" % (thread_block_size*block_size)),
            ]+[
            Assign("result[idx+%d]" % (o*thread_block_size),
                "op1[idx+%d] + op2[idx+%d]" % (
                    o*thread_block_size, 
                    o*thread_block_size))
            for o in range(block_size)
            ])
        )
    ])

mod = SourceModule(mod)

func = mod.get_function("add")
func(c_gpu, a_gpu, b_gpu, 
        block=(thread_block_size,1,1),
        grid=(macroblock_count,1))

c = cuda.from_device_like(c_gpu, a)

assert la.norm(c-(a+b)) == 0

