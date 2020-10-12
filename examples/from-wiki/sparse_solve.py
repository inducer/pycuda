#!python 
import pycuda.autoinit
import pycuda.driver as drv
import pycuda.gpuarray as gpuarray
import numpy
import numpy.linalg as la




def main_cg():
    from optparse import OptionParser

    parser = OptionParser(
            usage="%prog [options] MATRIX-MARKET-FILE")
    parser.add_option("-s", "--is-symmetric", action="store_true",
            help="Specify that the input matrix is already symmetric")
    options, args = parser.parse_args()

    from pycuda.tools import DeviceMemoryPool, PageLockedMemoryPool
    dev_pool = DeviceMemoryPool()
    pagelocked_pool = PageLockedMemoryPool()

    from scipy.io import mmread
    csr_mat = mmread(args[0]).tocsr().astype(numpy.float32)

    inv_mat_diag = 1/csr_mat.diagonal()

    print("building...")
    from pycuda.sparse.packeted import PacketedSpMV
    spmv = PacketedSpMV(csr_mat, options.is_symmetric, csr_mat.dtype)
    rhs = numpy.random.rand(spmv.shape[0]).astype(spmv.dtype)

    from pycuda.sparse.operator import DiagonalPreconditioner
    if True:
        precon = DiagonalPreconditioner(
                spmv.permute(gpuarray.to_gpu(
                    inv_mat_diag, allocator=dev_pool.allocate)))
    else:
        precon = None

    from pycuda.sparse.cg import solve_pkt_with_cg
    print("start solve")
    for i in range(4):
        start = drv.Event()
        stop = drv.Event()
        start.record()

        rhs_gpu = gpuarray.to_gpu(rhs, dev_pool.allocate)
        res_gpu, it_count, res_count = \
                solve_pkt_with_cg(spmv, rhs_gpu, precon,
                        tol=1e-7 if spmv.dtype == numpy.float64 else 5e-5,
                        pagelocked_allocator=pagelocked_pool.allocate)
        res = res_gpu.get()

        stop.record()
        stop.synchronize()

        elapsed = stop.time_since(start)*1e-3
        est_flops = (csr_mat.nnz*2*(it_count+res_count)
            + csr_mat.shape[0]*(2+2+2+2+2)*it_count)

        if precon is not None:
            est_flops += csr_mat.shape[0] * it_count

        print("residual norm: %g" % (la.norm(csr_mat*res - rhs)/la.norm(rhs)))
        print(("size: %d, elapsed: %g s, %d it, %d residual, it/second: %g, "
                "%g gflops/s" % (
                    csr_mat.shape[0],
                    elapsed, it_count, res_count, it_count/elapsed,
                    est_flops/elapsed/1e9)))

    # TODO: mixed precision
    # TODO: benchmark
    pagelocked_pool.stop_holding()
    dev_pool.stop_holding()





if __name__ == "__main__":
    print("starting...")
    main_cg()



