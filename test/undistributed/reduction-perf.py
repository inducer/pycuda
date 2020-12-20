import pycuda.autoinit
import pycuda.gpuarray as gpuarray
import pycuda.driver as cuda
import numpy


def main():
    from pytools import Table

    tbl = Table()
    tbl.add_row(("type", "size [MiB]", "time [ms]", "mem.bw [GB/s]"))

    for dtype_out in [numpy.float32, numpy.float64]:
        for ex in range(15, 27):
            sz = 1 << ex
            print(sz)

            from pycuda.curandom import rand as curand

            a_gpu = curand((sz,))
            b_gpu = curand((sz,))
            assert sz == a_gpu.shape[0]
            assert len(a_gpu.shape) == 1

            from pycuda.reduction import get_dot_kernel

            krnl = get_dot_kernel(dtype_out, a_gpu.dtype)

            elapsed = [0]

            def wrap_with_timer(f):
                def result(*args, **kwargs):
                    start = cuda.Event()
                    stop = cuda.Event()
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

            for i in range(cnt):
                krnl(
                    a_gpu,
                    b_gpu,
                    # krnl(a_gpu,
                    kernel_wrapper=wrap_with_timer,
                )

            bytes = a_gpu.nbytes * 2 * cnt
            secs = elapsed[0] * 1e-3

            tbl.add_row(
                (
                    str(dtype_out),
                    a_gpu.nbytes / (1 << 20),
                    elapsed[0] / cnt,
                    bytes / secs / 1e9,
                )
            )

    print(tbl)


if __name__ == "__main__":
    main()
