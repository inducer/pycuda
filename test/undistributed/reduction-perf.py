from __future__ import division
import pycuda.autoinit
import pycuda.gpuarray as gpuarray
import pycuda.driver as cuda



def main():
    from pytools import Table
    tbl = Table()
    tbl.add_row(("size [MiB]", "time [ms]", "mem.bw [GB/s]"))

    for ex in range(3,27):
        sz = 1 << ex
        print sz

        from pycuda.curandom import rand as curand
        a_gpu = curand((sz,))
        b_gpu = curand((sz,))

        from pycuda.reduction import get_sum_kernel, get_dot_kernel
        krnl = get_sum_kernel(a_gpu.dtype, a_gpu.dtype)
        #krnl = get_dot_kernel(a_gpu.dtype)

        elapsed = [0]

        def wrap_with_timer(f):
            def result(*args, **kwargs):
                start = cuda.Event()
                stop = cuda.Event()
                cuda.Context.synchronize()
                start.record()
                f(*args, **kwargs)
                stop.record()
                stop.synchronize()
                elapsed[0] += stop.time_since(start)

            return result

        # warm-up
        for i in range(3):
            krnl(a_gpu, b_gpu)

        cnt = 10
        #cnt = 1

        krnl.wrap_kernels(wrap_with_timer)
        for i in range(cnt):
            #krnl(a_gpu, b_gpu)
            krnl(a_gpu)

        bytes = a_gpu.nbytes*2*cnt
        secs = elapsed[0]*1e-3

        tbl.add_row((a_gpu.nbytes/(1<<20), elapsed[0]/cnt, bytes/secs/1e9))

    print tbl

if __name__ == "__main__":
    main()
