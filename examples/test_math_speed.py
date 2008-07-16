#! /usr/bin/env python
import pycuda.cumath as cuma
import pycuda._kernel as kernel
import pycuda.gpuarray as cuda
import pycuda.driver as drv
import types
import numpy as numpy
from pytools import Table

runs = 10

drv.init()
assert drv.Device.count() >= 1
ctx = drv.Device(0).make_context()


def time_cpu_execution(size,method,argumentCount):
    """times the execution time on the cpu"""

    start = drv.Event()
    end = drv.Event()
    start.record()

    a = numpy.zeros(size,numpy.float32)+1

    for x in range(runs):    
        for i in range(size):
            if argumentCount == 1:
                method(a[i])
            if argumentCount == 2:
                method(a[i],2)

    #stop timer
    end.record()
    end.synchronize()
        
    #calculate used time
    secs = start.time_till(end)

    return secs

def time_gpu_execution(size,method,argumentCount):
    """times the execution time on the gpu"""
    start = drv.Event()
    end = drv.Event()
    start.record()
    
    a = cuda.array(size)+1

    for x in range(runs):
        if argumentCount == 1:
            method(a)
        if argumentCount == 2:
            method(a,2)

    #stop timer
    end.record()
    end.synchronize()
        
    #calculate used time
    secs = start.time_till(end)

    return secs

#iterate over all methods and time the execution time with different array sizes
print "compile kernels"
kernel._compile_kernels(kernel)

#generate our output table, one for gpu, one for cpu
tblCPU = Table()
tblGPU = Table()
tblSPD = Table()

#contains all the method names
methods = ["size"]

for name in dir(cuma):
    if (name.startswith("__") and name.endswith("__")) == False:
        method = getattr(cuma, name)

        if type(method) == types.FunctionType:
            methods.append(name)
        
tblCPU.add_row(methods)
tblGPU.add_row(methods)
tblSPD.add_row(methods)

#generate arrays with differnt sizes
for power in range(1,20):
    size = 1<<power
    
    #temp variables
    rowCPU = [size]
    rowGPU = [size]
    rowSPD = [size]
          
    print "calculating: ", size
          
    for name in dir(cuma):
        if (name.startswith("__") and name.endswith("__")) == False:

            method = getattr(cuma, name)

            if type(method) == types.FunctionType:
                code = method.func_code
                argCount = code.co_argcount

                gpu_time = time_gpu_execution(size,method,argCount)
                cpu_time = time_cpu_execution(size,method,argCount)
                
                rowCPU.append(str(cpu_time/runs)[0:7])
                rowGPU.append(str(gpu_time/runs)[0:7])
                
                speed_cpu = size/(cpu_time/runs)
                speed_gpu = size/(gpu_time/runs)
                rowSPD.append(str(speed_gpu/speed_cpu)[0:7])
            
    tblCPU.add_row(rowCPU)
    tblGPU.add_row(rowGPU)
    tblSPD.add_row(rowSPD)
    
print ""

print "GPU Times (ms)"

print ""

print tblGPU

print ""

print "CPU Times (ms)"

print ""
print tblCPU


print ""

print "GPU VS CPU"

print ""
print tblSPD

