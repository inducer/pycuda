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

# begin graph capture, pull stream_2 into it as a dependency
#  https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#cross-stream-dependencies
stream_1 = cuda.Stream()
stream_2 = cuda.Stream()
stream_1.begin_capture()
event_init = cuda.Event()
event_a = cuda.Event()
event_b = cuda.Event()

event_init.record(stream_1)
stream_2.wait_for_event(event_init)

cuda.memcpy_htod_async(a_gpu, a, stream_1)
func_plus(a_gpu, numpy.int32(2), block=(4, 4, 1), stream=stream_1)
event_a.record(stream_1)

cuda.memcpy_htod_async(b_gpu, b, stream_2)
func_plus(b_gpu, numpy.int32(3), block=(4, 4, 1), stream=stream_2)
event_b.record(stream_2)

stream_1.wait_for_event(event_a)
stream_1.wait_for_event(event_b)
func_times(a_gpu, b_gpu, block=(4, 4, 1), stream=stream_1)
cuda.memcpy_dtoh_async(result, a_gpu, stream_1)

graph = stream_1.end_capture()
graph.debug_dot_print("test.dot")  # print dotfile of graph
instance = graph.instance()

# using a separate graph stream to launch, this is not strictly necessary
stream_graph = cuda.Stream()
instance.launch(stream_graph)

print("original arrays:")
print(a)
print(b)
print("(0+2)x(0+3) = 6, using a kernel graph of 3 kernels:")
print(result)
