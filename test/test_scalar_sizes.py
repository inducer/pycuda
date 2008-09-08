from __future__ import division




def time_grid(size, block_size, block_count):
    import pycuda.gpuarray as gpuarray
    import numpy
    data = gpuarray.zeros((size,), dtype=numpy.float32)
    from pycuda._kernel import get_axpbyz_kernel
    kernel = get_axpbyz_kernel()

    from time import time
    start_time = time()

    import pycuda.driver as cuda

    count = 10
    cuda.Context.synchronize()
    for i in range(count):
        kernel.prepared_call((block_count,1), 
            2, data.gpudata,
            2, data.gpudata,
            data.gpudata, size)
    cuda.Context.synchronize()

    return (time()-start_time)/count





def main():
    import pycuda.autoinit
    import pycuda.driver as cuda
    from pycuda.tools import DeviceData, OccupancyRecord
    import sys

    devdata = DeviceData(pycuda.autoinit.device)
    max_warps = devdata.max_threads//devdata.warp_size

    for size_exp in range(15, 25):
        size = 2**size_exp

        for rep in range(2):
            times = []
            for warp_count in range(1, max_warps+1):
                block_size = warp_count*devdata.warp_size
                max_block_count = min(
                        (size+block_size-1) // block_size,
                        pycuda.autoinit.device.get_attribute(
                            cuda.device_attribute.MAX_GRID_DIM_X)
                        )
                block_step = 1
                while max_block_count // block_step > 128:
                    block_step *= 2

                for block_count in range(1, max_block_count+1, block_step):
                    times.append((
                        time_grid(size, block_size, block_count), 
                        warp_count, block_count))
                    sys.stdout.write(".")
                    sys.stdout.flush()
            sys.stdout.write("\n")
            sys.stdout.flush()

            times.sort()
            print "size=%d" % size
            from pytools import Table
            tbl = Table()
            tbl.add_row(("time", "warps", "blocks", "occupancy"))
            for t, warp_count, block_count in times[:10]:
                tbl.add_row(("%.7g" % t, warp_count, block_count,
                    OccupancyRecord(devdata, warp_count*devdata.warp_size).occupancy
                    ))
            print tbl

            from pycuda.gpuarray import splay_old, splay
            (bc, _), (tpb, _, _) = splay_old(size)
            print  "old splay:", time_grid(size, tpb, bc), tpb//devdata.warp_size, bc
            (bc, _), (tpb, _, _) = splay(size)
            print  "new splay:", time_grid(size, tpb, bc), tpb//devdata.warp_size, bc

        raw_input()




if __name__ == "__main__":
    main()
