import pycuda.driver as cuda
import pycuda.autoinit
import numpy
import numpy.linalg as la

block_size = 16
thread_block_size = 32
macroblock_count = 33

a = numpy.random.randn(block_size*thread_block_size*macroblock_count)\
        .astype(numpy.float32)
b = numpy.random.randn(block_size*thread_block_size*macroblock_count)\
        .astype(numpy.float32)

a_gpu = cuda.to_device(a)
b_gpu = cuda.to_device(b)
c_gpu = cuda.mem_alloc(a.nbytes)

from jinja2 import Template

tpl = Template("""
    typedef {{ type_name }} value_type;

    __global__ void add(value_type *result, value_type *op1, value_type *op2)
    {
      int idx = threadIdx.x + {{ thread_block_size }} * {{block_size}} * blockIdx.x;

      #for i in range(block_size)
          #set offset = i*thread_block_size
          result[idx + {{ offset }}] = op1[idx + {{ offset }}] + op2[idx + {{ offset }}];
      #endfor
    }
    """,
    line_statement_prefix="#")

rendered_tpl = tpl.render(
    type_name="float",
    block_size=block_size,
    thread_block_size=thread_block_size)

mod = cuda.SourceModule(rendered_tpl)

func = mod.get_function("add")
func(c_gpu, a_gpu, b_gpu, 
        block=(thread_block_size,1,1),
        grid=(macroblock_count,1))

c = cuda.from_device_like(c_gpu, a)

assert la.norm(c-(a+b)) == 0
