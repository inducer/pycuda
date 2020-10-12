#!python 
#! /usr/bin/env python
# A simple program to illustrate kernel concurrency with PyCuda.
# Reference: Chapter 3.2.6.5 in Cuda C Programming Guide Version 3.2.
# Jesse Lu, 2011-04-04

import numpy as np
import pycuda.autoinit
import pycuda.driver as drv
from pycuda.compiler import SourceModule

#
# Set up test scenario.
# 

# Create a simple test kernel.
mod = SourceModule("""
__global__ void my_kernel(float *d) {
    const int i = threadIdx.x;
    for (int m=0; m<100; m++) {
        for (int k=0; k<100 ; k++)
            d[i] = d[i] * 2.0;
        for (int k=0; k<100 ; k++)
            d[i] = d[i] / 2.0;
    }
    d[i] = d[i] * 2.0;
}
""")
my_kernel = mod.get_function("my_kernel")

# Create the test data on the host.
N = 400 # Size of datasets.
n = 2 # Number of datasets (and concurrent operations) used.
data, data_check, d_data = [], [], []
for k in range(n):
    data.append(np.random.randn(N).astype(np.float32)) # Create random data.
    data_check.append(data[k].copy()) # For checking the result afterwards. 
    d_data.append(drv.mem_alloc(data[k].nbytes)) # Allocate memory on device.

#
# Start concurrency test.
#

# Use this event as a reference point.
ref = drv.Event()
ref.record()

# Create the streams and events needed.
stream, event = [], []
marker_names = ['kernel_begin', 'kernel_end']
for k in range(n):
    stream.append(drv.Stream())
    event.append({marker_names[l]: drv.Event() for l in range(len(marker_names))})

# Transfer to device.
for k in range(n):
    drv.memcpy_htod(d_data[k], data[k]) 

# Run kernels many times, we will only keep data from last loop iteration.
for j in range(10):
    for k in range(n):
        event[k]['kernel_begin'].record(stream[k])
        my_kernel(d_data[k], block=(N,1,1), stream=stream[k]) 
    for k in range(n): # Commenting out this line should break concurrency.
        event[k]['kernel_end'].record(stream[k])

# Transfer data back to host.
for k in range(n):
    drv.memcpy_dtoh(data[k], d_data[k]) 

# 
# Output results.
#

print('\n=== Device attributes')
dev = pycuda.autoinit.device
print(('Name:', dev.name()))
print(('Compute capability:', dev.compute_capability()))
print(('Concurrent Kernels:', \
    bool(dev.get_attribute(drv.device_attribute.CONCURRENT_KERNELS))))

print('\n=== Checking answers')
for k in range(n):
    print(('Dataset', k, ':',))
    if (np.linalg.norm((data_check[k] * 2**(j+1)) - data[k]) == 0.0):
        print('passed.')
    else:
        print('FAILED!')

print('\n=== Timing info (for last set of kernel launches)')
for k in range(n):
    print(('Dataset', k)) 
    for l in range(len(marker_names)):
        print((marker_names[l], ':', ref.time_till(event[k][marker_names[l]])))

