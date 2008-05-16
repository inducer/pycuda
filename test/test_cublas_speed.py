import pycuda.driver as drv
import numpy
import numpy.linalg as la




def main():
    drv.init()
    assert drv.Device.count() >= 1
    ctx = drv.Device(0).make_context()

    import pycuda.blas as blas

    sizes = []
    times = []
    flops = []
    for power in range(10, 24): # 24
        size = 1<<power
        print size
        sizes.append(size)
        a = 10*blas.ones((size,), dtype=numpy.float32)
        b = 33*blas.ones((size,), dtype=numpy.float32)

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

    if True:
        from matplotlib.pylab import semilogx, show, title
        title("time to add two vectors")
        semilogx(sizes, times)
        show()
        title("flops")
        semilogx(sizes, [f/t for f, t in zip(flops,times)])
        show()


        




if __name__ == "__main__":
    main()

