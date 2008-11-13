#! /usr/bin/env python
import pycuda.driver as drv
import numpy
import numpy.linalg as la




def main():
    drv.init()
    assert drv.Device.count() >= 1
    ctx = drv.Device(0).make_context()

    import pycuda.gpuarray as gpuarray

    sizes = []
    times_gpu = []
    flops_gpu = []
    flops_cpu = []
    times_cpu = []
    
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

        # gpu -----------------------------------------------------------------
        start = drv.Event()
        end = drv.Event()
        start.record()

        for i in range(count):
            a+b
            
        end.record()
        end.synchronize()
        
        secs = start.time_till(end)*1e-3

        times_gpu.append(secs/count)
        flops_gpu.append(size)

        # cpu -----------------------------------------------------------------
        a_cpu = numpy.random.randn(size).astype(numpy.float32)
        b_cpu = numpy.random.randn(size).astype(numpy.float32)

        #start timer
        from time import time
        start = time()
        for i in range(count):
            a_cpu + b_cpu
        secs = time() - start

        times_cpu.append(secs/count)
        flops_cpu.append(size)
            
            
    # calculate pseudo flops
    flops_gpu = [f/t for f, t in zip(flops_gpu,times_gpu)]
    flops_cpu = [f/t for f, t in zip(flops_cpu,times_cpu)]

    from pytools import Table
    tbl = Table()
    tbl.add_row(("Size", "Time GPU", "Size/Time GPU", 
        "Time CPU","Size/Time CPU","GPU vs CPU speedup"))
    for s, t, f, t_cpu, f_cpu in zip(sizes, times_gpu, flops_gpu, times_cpu, flops_cpu):
        tbl.add_row((s, t, f, t_cpu, f_cpu, f/f_cpu))
    print tbl
        




if __name__ == "__main__":
    main()
