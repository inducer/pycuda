# Sample source code from the Tutorial Introduction in the documentation.
import pycuda.driver as cuda
import pycuda.autoinit  # noqa
from pycuda.compiler import SourceModule

mod = SourceModule("""
    __global__ void plus(float *a, int num)
    {
      int idx = threadIdx.x + threadIdx.y*4;
      a[idx] += num;
    }
    __global__ void times(float *a, float *b)
    {
      int idx = threadIdx.x + threadIdx.y*4;
      a[idx] *= b[idx];
    }
    """)
func_plus = mod.get_function("plus")
func_times = mod.get_function("times")

import numpy
a = numpy.zeros((4, 4)).astype(numpy.float32)
a_gpu = cuda.mem_alloc_like(a)
b = numpy.zeros((4, 4)).astype(numpy.float32)
b_gpu = cuda.mem_alloc_like(b)
result = numpy.zeros_like(b)
b2_gpu = cuda.mem_alloc_like(b)

stream_1 = cuda.Stream()
stream_1.begin_capture()
cuda.memcpy_htod_async(a_gpu, a, stream_1)
cuda.memcpy_htod_async(b_gpu, b, stream_1)
cuda.memcpy_htod_async(b2_gpu, b, stream_1)
func_plus(a_gpu, numpy.int32(2), block=(4, 4, 1), stream=stream_1)
_, _, graph, deps = stream_1.get_capture_info_v2()
first_node = graph.add_kernel_node(b_gpu, numpy.int32(3), block=(4, 4, 1), func=func_plus, dependencies=deps)
stream_1.update_capture_dependencies([first_node], 1)

_, _, graph, deps = stream_1.get_capture_info_v2()
second_node = graph.add_kernel_node(a_gpu, b_gpu, block=(4, 4, 1), func=func_times, dependencies=deps)
stream_1.update_capture_dependencies([second_node], 1)
cuda.memcpy_dtoh_async(result, a_gpu, stream_1)

graph = stream_1.end_capture()
graph.debug_dot_print("test.dot")  # print dotfile of graph
instance = graph.instantiate()

# Setting dynamic parameters
instance.kernel_node_set_params(b2_gpu, numpy.int32(100), block=(4, 4, 1), func=func_plus, kernel_node=first_node)
instance.kernel_node_set_params(a_gpu, b2_gpu, block=(4, 4, 1), func=func_times, kernel_node=second_node)
instance.launch()

print("original arrays:")
print(a)
print(b)
print("(0+2)x(0+100) = 200, using a kernel graph of 3 kernels:")
print(result)