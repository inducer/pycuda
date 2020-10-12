#!python 
# SimpleSpeedTest.py

# Very simple speed testing code
# Shows you how to run a loop over sin() using different methods
# with a note of the time each method takes
# For the GPU this uses SourceModule, ElementwiseKernel, GPUArray
# For the CPU this uses numpy
# Ian@IanOzsvald.com

# Using a WinXP Intel Core2 Duo 2.66GHz CPU (1 CPU used)
# with a 9800GT GPU I get the following timings (smaller is better):
#
# Using nbr_values == 8192
# Calculating 100000 iterations
# SourceModule time and first three results:
# 0.166590s, [ 0.005477  0.005477  0.005477]
# Elementwise time and first three results:
# 0.171657s, [ 0.005477  0.005477  0.005477]
# Elementwise Python looping time and first three results:
# 1.487470s, [ 0.005477  0.005477  0.005477]
# GPUArray time and first three results:
# 4.740007s, [ 0.005477  0.005477  0.005477]
# CPU time and first three results:
# 32.933660s, [ 0.005477  0.005477  0.005477]
#
# 
# Using Win 7 x64, GTX 470 GPU, X5650 Xeon,
# Driver v301.42, CUDA 4.2, Python 2.7 x64,
# PyCuda 2012.1 gave the following results:
#
# Using nbr_values == 8192
# Calculating 100000 iterations
# SourceModule time and first three results:
# 0.058321s, [ 0.005477  0.005477  0.005477]
# Elementwise time and first three results:
# 0.102110s, [ 0.005477  0.005477  0.005477]
# Elementwise Python looping time and first three results:
# 2.428810s, [ 0.005477  0.005477  0.005477]
# GPUArray time and first three results:
# 8.421861s, [ 0.005477  0.005477  0.005477]
# CPU time measured using :
# 5.905661s, [ 0.005477  0.005477  0.005477]


import pycuda.driver as drv
import pycuda.tools
import pycuda.autoinit
import numpy
from pycuda.compiler import SourceModule
import pycuda.gpuarray as gpuarray
import pycuda.cumath
from pycuda.elementwise import ElementwiseKernel

blocks = 64
block_size = 128
nbr_values = blocks * block_size

print("Using nbr_values ==", nbr_values)

# Number of iterations for the calculations,
# 100 is very quick, 2000000 will take a while
n_iter = 100000
print("Calculating %d iterations" % (n_iter))

# create two timers so we can speed-test each approach
start = drv.Event()
end = drv.Event()

######################
# SourceModele SECTION
# We write the C code and the indexing and we have lots of control

mod = SourceModule("""
__global__ void gpusin(float *dest, float *a, int n_iter)
{
  const int i = blockDim.x*blockIdx.x + threadIdx.x;
  for(int n = 0; n < n_iter; n++) {
    a[i] = sin(a[i]);
  }
  dest[i] = a[i];
}
""")

gpusin = mod.get_function("gpusin")

# create an array of 1s
a = numpy.ones(nbr_values).astype(numpy.float32)
# create a destination array that will receive the result
dest = numpy.zeros_like(a)

start.record() # start timing
gpusin(drv.Out(dest), drv.In(a), numpy.int32(n_iter), grid=(blocks,1), block=(block_size,1,1) )
end.record() # end timing
# calculate the run length
end.synchronize()
secs = start.time_till(end)*1e-3
print("SourceModule time and first three results:")
print("%fs, %s" % (secs, str(dest[:3])))


#####################
# Elementwise SECTION
# use an ElementwiseKernel with sin in a for loop all in C call from Python
kernel = ElementwiseKernel(
   "float *a, int n_iter",
   "for(int n = 0; n < n_iter; n++) { a[i] = sin(a[i]);}",
   "gpusin")

a = numpy.ones(nbr_values).astype(numpy.float32)
a_gpu = gpuarray.to_gpu(a)
start.record() # start timing
kernel(a_gpu, numpy.int(n_iter))
end.record() # end timing
# calculate the run length
end.synchronize()
secs = start.time_till(end)*1e-3
print("Elementwise time and first three results:")
print("%fs, %s" % (secs, str(a_gpu.get()[:3])))


####################################
# Elementwise Python looping SECTION
# as Elementwise but the for loop is in Python, not in C
kernel = ElementwiseKernel(
   "float *a",
   "a[i] = sin(a[i]);",
   "gpusin")

a = numpy.ones(nbr_values).astype(numpy.float32)
a_gpu = gpuarray.to_gpu(a)
start.record() # start timing
for i in range(n_iter):
    kernel(a_gpu)
end.record() # end timing
# calculate the run length
end.synchronize()
secs = start.time_till(end)*1e-3
print("Elementwise Python looping time and first three results:")
print("%fs, %s" % (secs, str(a_gpu.get()[:3])))


##################
# GPUArray SECTION
# The result is copied back to main memory on each iteration, this is a bottleneck

a = numpy.ones(nbr_values).astype(numpy.float32)
a_gpu = gpuarray.to_gpu(a)
start.record() # start timing
for i in range(n_iter):
    a_gpu = pycuda.cumath.sin(a_gpu)
end.record() # end timing
# calculate the run length
end.synchronize()
secs = start.time_till(end)*1e-3
print("GPUArray time and first three results:")
print("%fs, %s" % (secs, str(a_gpu.get()[:3])))


#############
# CPU SECTION
# use numpy the calculate the result on the CPU for reference

a = numpy.ones(nbr_values).astype(numpy.float32)
start.record() # start timing
start.synchronize()

for i in range(n_iter):
    a = numpy.sin(a)

end.record() # end timing
# calculate the run length
end.synchronize()
secs = start.time_till(end)*1e-3
print("CPU time and first three results:")
print("%fs, %s" % (secs, str(a[:3])))

