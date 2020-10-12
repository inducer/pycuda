#!python 
#! /usr/bin/env python
# DO NOT USE THIS AS A BENCHMARK. See
# http://documen.tician.de/pycuda/array.html#pycuda.curandom.rand

import pycuda.autoinit
import pycuda.driver as drv
import pycuda.curandom as curandom
import numpy
import numpy.linalg as la
from pytools import Table




def main():
    import pycuda.gpuarray as gpuarray

    sizes = []
    times = []
    flops = []
    flopsCPU = []
    timesCPU = []

    for power in range(10, 25): # 24
        size = 1<<power
        print(size)
        sizes.append(size)
        a = gpuarray.zeros((size,), dtype=numpy.float32)

        if power > 20:
            count = 100
        else:
            count = 1000

        #start timer
        start = drv.Event()
        end = drv.Event()
        start.record()

        #cuda operation which fills the array with random numbers
        for i in range(count):
            curandom.rand((size, ))

        #stop timer
        end.record()
        end.synchronize()

        #calculate used time
        secs = start.time_till(end)*1e-3

        times.append(secs/count)
        flops.append(size)

        #cpu operations which fills teh array with random data
        a = numpy.array((size,), dtype=numpy.float32)

        #start timer
        start = drv.Event()
        end = drv.Event()
        start.record()

        #cpu operation which fills the array with random data
        for i in range(count):
            numpy.random.rand(size).astype(numpy.float32)

        #stop timer
        end.record()
        end.synchronize()

        #calculate used time
        secs = start.time_till(end)*1e-3

        #add results to variable
        timesCPU.append(secs/count)
        flopsCPU.append(size)


    #calculate pseudo flops
    flops = [f/t for f, t in zip(flops,times)]
    flopsCPU = [f/t for f, t in zip(flopsCPU,timesCPU)]

    #print the data out
    tbl = Table()
    tbl.add_row(("Size", "Time GPU", "Size/Time GPU", "Time CPU","Size/Time CPU","GPU vs CPU speedup"))
    for s, t, f,tCpu,fCpu in zip(sizes, times, flops,timesCPU,flopsCPU):
        tbl.add_row((s,t,f,tCpu,fCpu,f/fCpu))
    print(tbl)


if __name__ == "__main__":
    main()

