from __future__ import absolute_import
import pycuda.driver as cuda
import pycuda.autoinit
import numpy
import numpy.linalg as la
from pycuda.compiler import SourceModule

thread_strides = 16
block_size = 32
macroblock_count = 33

total_size = thread_strides*block_size*macroblock_count
dtype = numpy.float32

a = numpy.random.randn(total_size).astype(dtype)
b = numpy.random.randn(total_size).astype(dtype)

a_gpu = cuda.to_device(a)
b_gpu = cuda.to_device(b)
c_gpu = cuda.mem_alloc(a.nbytes)

from jinja2 import Template

tpl = Template("""
    __global__ void add(
            {{ type_name }} *tgt, 
            {{ type_name }} *op1, 
            {{ type_name }} *op2)
    {
      int idx = threadIdx.x + 
        {{ block_size }} * {{thread_strides}}
        * blockIdx.x;

      {% for i in range(thread_strides) %}
          {% set offset = i*block_size %}
          tgt[idx + {{ offset }}] = 
            op1[idx + {{ offset }}] 
            + op2[idx + {{ offset }}];
      {% endfor %}
    }""")

rendered_tpl = tpl.render(
    type_name="float", thread_strides=thread_strides,
    block_size=block_size)

mod = SourceModule(rendered_tpl)
# end

func = mod.get_function("add")
func(c_gpu, a_gpu, b_gpu, 
        block=(block_size,1,1),
        grid=(macroblock_count,1))

c = cuda.from_device_like(c_gpu, a)

assert la.norm(c-(a+b)) == 0
