#! /usr/bin/env python
import pycuda.driver as drv
import numpy
import numpy.linalg as la




def main():
    drv.init()
    assert drv.Device.count() >= 1
    ctx = drv.Device(0).make_context()

    import pycuda.gpuarray as gpuarray

    # make sure all the kernels are compiled
    gpuarray.GPUArray.compile_kernels()
    print "done compiling"

    sizes = []
    times = []
    flops = []
    flopsCPU = []
    timesCPU = []
    
    for power in range(10, 25): # 24
        size = 1<<power
        print size
        sizes.append(size)
        a = gpuarray.zeros((size,), dtype=numpy.float32)
        b = gpuarray.zeros((size,), dtype=numpy.float32)
        b.fill(1)

        if power > 20:
            count = 100
        else:
            count = 1000

        #start timer
        start = drv.Event()
        end = drv.Event()
        start.record()

        #cuda operation which adds two arrays over count time to get an average
        for i in range(count):
            a+b
            
        #stop timer
        end.record()
        end.synchronize()
        
        #calculate used time
        secs = start.time_till(end)*1e-3

        times.append(secs/count)
        flops.append(size)

        #cpu operations which adds two arrays
        aCpu = numpy.random.randn(size).astype(numpy.float32)
        bCpu = numpy.random.randn(size).astype(numpy.float32)

        #start timer
        start = drv.Event()
        end = drv.Event()
        start.record()

        #cpu operation which adds two arrays over count time to get an average        
        for i in range(count):
            aCpu + bCpu

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
    try:
        from matplotlib.pylab import semilogx, show, title
    except ImportError:
        from pytools import Table
        tbl = Table()
        tbl.add_row(("Size", "Time GPU", "Size/Time GPU", "Time CPU","Size/Time CPU","GPU vs CPU speedup"))
        for s, t, f,tCpu,fCpu in zip(sizes, times, flops,timesCPU,flopsCPU):
            tbl.add_row((s,t,f,tCpu,fCpu,f/fCpu))
        print tbl
    else:
        title("time to add two vectors")
        semilogx(sizes, times)
        show()
        title("flops")
        semilogx(sizes, flops)
        show()



        




if __name__ == "__main__":
    main()
