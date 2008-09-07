



def main():
    import pycuda.autoinit
    import pycuda.driver as cuda
    from pycuda.tools import DeviceData, OccupancyRecord
    import pycuda.gpuarray as gpuarray
    import numpy

    devdata = DeviceData(pycuda.autoinit.device)
    max_warps = devdata.max_threads/devdata.warp_size

    from pycuda._kernel import get_axpbyz_kernel
    kernel = get_axpbyz_kernel()
    for size_exp in range(8, 25):
        size = 2**size_exp
        data = gpuarray.zeros((size,), dtype=numpy.float32)

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
                kernel.set_block_shape(block_size, 1, 1)
                t = min(
                        kernel.prepared_timed_call((block_count,1), 
                            2, data.gpudata,
                            2, data.gpudata,
                            data.gpudata, size)
                        for i in range(10))
                times.append((t, warp_count, block_count))

        times.sort()
        print "size=%d" % size
        from pytools import Table
        tbl = Table()
        tbl.add_row(("time", "warps", "blocks", "occupancy"))
        for t, warp_count, block_count in times[:20]:
            tbl.add_row(("%.7g" % t, warp_count, block_count,
                OccupancyRecord(devdata, warp_count*devdata.warp_size).occupancy
                ))
        print tbl

        raw_input()




if __name__ == "__main__":
    main()
