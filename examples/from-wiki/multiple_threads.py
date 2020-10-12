#!python 
# Derived from a test case by Chris Heuser
# Also see FAQ about PyCUDA and threads.


import pycuda
import pycuda.driver as cuda
from pycuda.compiler import SourceModule
import threading
import numpy

class GPUThread(threading.Thread):
    def __init__(self, number, some_array):
        threading.Thread.__init__(self)

        self.number = number
        self.some_array = some_array

    def run(self):
        self.dev = cuda.Device(self.number)
        self.ctx = self.dev.make_context()

        self.array_gpu = cuda.mem_alloc(some_array.nbytes)
        cuda.memcpy_htod(self.array_gpu, some_array)

        test_kernel(self.array_gpu)
        print("successful exit from thread %d" % self.number)
        self.ctx.pop()

        del self.array_gpu
        del self.ctx

def test_kernel(input_array_gpu):
    mod = SourceModule("""
        __global__ void f(float * out, float * in)
        {
            int idx = threadIdx.x;
            out[idx] = in[idx] + 6;
        }
        """)
    func = mod.get_function("f")

    output_array = numpy.zeros((1,512))
    output_array_gpu = cuda.mem_alloc(output_array.nbytes)

    func(output_array_gpu,
          input_array_gpu,
          block=(512,1,1))
    cuda.memcpy_dtoh(output_array, output_array_gpu)

    return output_array

cuda.init()
some_array = numpy.ones((1,512), dtype=numpy.float32)
num = cuda.Device.count()

gpu_thread_list = []
for i in range(num):
    gpu_thread = GPUThread(i, some_array)
    gpu_thread.start()
    gpu_thread_list.append(gpu_thread)


