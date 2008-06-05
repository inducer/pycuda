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
    for power in range(10, 24): # 24
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

        start = drv.Event()
        end = drv.Event()
        start.record()
        for i in range(count):
            a+b
        end.record()
        end.synchronize()
        secs = start.time_till(end)*1e-3

        times.append(secs/count)
        flops.append(size*4)

    flops = [f/t for f, t in zip(flops,times)]
    try:
        from matplotlib.pylab import semilogx, show, title
    except ImportError:
        from pytools import Table
        tbl = Table()
        tbl.add_row(("Size", "Time", "Flops"))
        for s, t, f in zip(sizes, times, flops):
            tbl.add_row((s,t,f))
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
