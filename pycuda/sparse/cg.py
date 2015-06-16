from __future__ import division
from __future__ import absolute_import
from pycuda.sparse.inner import AsyncInnerProduct
from pytools import memoize_method
import pycuda.gpuarray as gpuarray

import numpy as np




class ConvergenceError(RuntimeError):
    pass



class CGStateContainer:
    def __init__(self, operator, precon=None, pagelocked_allocator=None):
        if precon is None:
            from pycuda.sparse.operator import IdentityOperator
            precon = IdentityOperator(operator.dtype, operator.shape[0])

        self.operator = operator
        self.precon = precon

        self.pagelocked_allocator = pagelocked_allocator

    @memoize_method
    def make_lc2_kernel(self, dtype, a_is_gpu, b_is_gpu):
        from pycuda.elementwise import get_linear_combination_kernel
        return get_linear_combination_kernel((
                (a_is_gpu, dtype, dtype),
                (b_is_gpu, dtype, dtype)
                ), dtype)

    def lc2(self, a, x, b, y, out=None):
        if out is None:
            out = gpuarray.empty(x.shape, dtype=x.dtype,
                    allocator=x.allocator)

        assert x.dtype == y.dtype == out.dtype
        a_is_gpu = isinstance(a, gpuarray.GPUArray)
        b_is_gpu = isinstance(b, gpuarray.GPUArray)
        assert x.shape == y.shape == out.shape

        kernel, texrefs = self.make_lc2_kernel(
                x.dtype, a_is_gpu, b_is_gpu)

        texrefs = texrefs[:]

        args = []

        if a_is_gpu:
            assert a.dtype == x.dtype
            assert a.shape == ()
            a.bind_to_texref_ext(texrefs.pop(0), allow_double_hack=True)
        else:
            args.append(a)
        args.append(x.gpudata)

        if b_is_gpu:
            assert b.dtype == y.dtype
            assert b.shape == ()
            b.bind_to_texref_ext(texrefs.pop(0), allow_double_hack=True)
        else:
            args.append(b)
        args.append(y.gpudata)
        args.append(out.gpudata)
        args.append(x.mem_size)

        kernel.prepared_call(x._grid, x._block, *args)

        return out

    @memoize_method
    def guarded_div_kernel(self, dtype_x, dtype_y, dtype_z):
        from pycuda.elementwise import get_elwise_kernel
        from pycuda.tools import dtype_to_ctype
        return get_elwise_kernel(
                "%(tp_x)s *x, %(tp_y)s *y, %(tp_z)s *z" % {
                    "tp_x": dtype_to_ctype(dtype_x),
                    "tp_y": dtype_to_ctype(dtype_y),
                    "tp_z": dtype_to_ctype(dtype_z),
                    },
                "z[i] = y[i] == 0 ? 0 : (x[i] / y[i])",
                "divide")

    def guarded_div(self, a, b):
        from pycuda.gpuarray import _get_common_dtype
        result = a._new_like_me(_get_common_dtype(a, b))

        assert a.shape == b.shape

        func = self.guarded_div_kernel(a.dtype, b.dtype, result.dtype)
        func.prepared_async_call(a._grid, a._block, None,
                a.gpudata, b.gpudata,
                result.gpudata, a.mem_size)

        return result

    def reset(self, rhs, x=None):
        self.rhs = rhs

        if x is None:
            x = np.zeros((self.operator.shape[0],))
        self.x = x

        self.residual = rhs - self.operator(x)

        self.d = self.precon(self.residual)

        # grows at the end
        delta = AsyncInnerProduct(self.residual, self.d,
                self.pagelocked_allocator)
        self.real_delta_queue = [delta]
        self.delta = delta.gpu_result

    def one_iteration(self, compute_real_residual=False):
        # typed up from J.R. Shewchuk,
        # An Introduction to the Conjugate Gradient Method
        # Without the Agonizing Pain, Edition 1 1/4 [8/1994]
        # Appendix B3

        q = self.operator(self.d)
        myip = gpuarray.dot(self.d, q)
        alpha = self.guarded_div(self.delta, myip)

        self.lc2(1, self.x, alpha, self.d, out=self.x)

        if compute_real_residual:
            self.residual = self.lc2(
                    1, self.rhs, -1, self.operator(self.x))
        else:
            self.lc2(1, self.residual, -alpha, q, out=self.residual)

        s = self.precon(self.residual)
        delta_old = self.delta
        delta = AsyncInnerProduct(self.residual, s,
                self.pagelocked_allocator)
        self.delta = delta.gpu_result
        beta = self.guarded_div(self.delta, delta_old)

        self.lc2(1, s, beta, self.d, out=self.d)

        if compute_real_residual:
            self.real_delta_queue.append(delta)

    def run(self, max_iterations=None, tol=1e-7, debug_callback=None):
        check_interval = 20

        if max_iterations is None:
            max_iterations = max(
                    3*check_interval+1, 10 * self.operator.shape[0])
        real_resid_interval = min(self.operator.shape[0], 50)

        iterations = 0
        delta_0 = None
        while iterations < max_iterations:
            compute_real_residual = \
                    iterations % real_resid_interval == 0

            self.one_iteration(
                    compute_real_residual=compute_real_residual)

            if debug_callback is not None:
                if compute_real_residual:
                    what = "it+residual"
                else:
                    what = "it"

                debug_callback(what, iterations, self.x,
                        self.residual, self.d, self.delta)

            # do often enough to allow AsyncInnerProduct
            # to progress through (polled) event chain
            rdq = self.real_delta_queue
            if iterations % check_interval == 0:
                if delta_0 is None:
                    delta_0 = rdq[0].get_host_result()
                    if delta_0 is not None:
                        rdq.pop(0)

                if delta_0 is not None:
                    i = 0
                    while i < len(rdq):
                        delta = rdq[i].get_host_result()
                        if delta is not None:
                            if abs(delta) < tol*tol * abs(delta_0):
                                if debug_callback is not None:
                                    debug_callback("end", iterations,
                                            self.x, self.residual,
                                            self.d, self.delta)
                                return self.x
                            rdq.pop(i)
                        else:
                            i += 1

            iterations += 1

        raise ConvergenceError("cg failed to converge")




def solve_pkt_with_cg(pkt_spmv, b, precon=None, x=None, tol=1e-7, max_iterations=None,
        debug=False, pagelocked_allocator=None):
    if x is None:
        x = gpuarray.zeros(pkt_spmv.shape[0], dtype=pkt_spmv.dtype,
                allocator=b.allocator)
    else:
        x = pkt_spmv.permute(x)

    if pagelocked_allocator is None:
        pagelocked_allocator = drv.pagelocked_empty

    cg = CGStateContainer(pkt_spmv, precon,
            pagelocked_allocator=pagelocked_allocator)

    cg.reset(pkt_spmv.permute(b), x)

    it_count = [0]
    res_count = [0]
    def debug_callback(what, it_number, x, resid, d, delta):
        if what == "it":
            it_count[0] += 1
        elif what == "it+residual":
            res_count[0] += 1
            it_count[0] += 1

    result = cg.run(max_iterations, tol,
            debug_callback=debug_callback)

    return pkt_spmv.unpermute(result), it_count[0], res_count[0]




