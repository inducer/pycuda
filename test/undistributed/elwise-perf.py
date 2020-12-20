import pycuda.driver as drv
import pycuda.autoinit
import numpy
import numpy.linalg as la


def main():
    from pytools import Table

    tbl = Table()
    tbl.add_row(("size [MiB]", "time [s]", "mem.bw [GB/s]"))

    import pycuda.gpuarray as gpuarray

    # they're floats, i.e. 4 bytes each
    for power in range(10, 28):
        size = 1 << power
        print(size)

        a = gpuarray.empty((size,), dtype=numpy.float32)
        b = gpuarray.empty_like(a)
        a.fill(1)
        b.fill(2)

        if power > 20:
            count = 10
        else:
            count = 100

        elapsed = [0]

        def add_timer(_, time):
            elapsed[0] += time()

        for i in range(count):
            a.mul_add(1, b, 2, add_timer)

        bytes = a.nbytes * count * 3
        bytes = a.nbytes * count * 3

        tbl.add_row(
            (a.nbytes / (1 << 20), elapsed[0] / count, bytes / elapsed[0] / 1e9)
        )

    print(tbl)


if __name__ == "__main__":
    main()
