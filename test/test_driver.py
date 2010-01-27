from __future__ import division
import numpy
import numpy.linalg as la
from pycuda.tools import mark_cuda_test




def have_pycuda():
    try:
        import pycuda
        return True
    except:
        return False


if have_pycuda():
    import pycuda.gpuarray as gpuarray
    import pycuda.driver as drv
    from pycuda.compiler import SourceModule




class TestDriver:
    disabled = not have_pycuda()

    @mark_cuda_test
    def test_memory(self):
        z = numpy.random.randn(400).astype(numpy.float32)
        new_z = drv.from_device_like(drv.to_device(z), z)
        assert la.norm(new_z-z) == 0

    @mark_cuda_test
    def test_simple_kernel(self):
        mod = SourceModule("""
        __global__ void multiply_them(float *dest, float *a, float *b)
        {
          const int i = threadIdx.x;
          dest[i] = a[i] * b[i];
        }
        """)

        multiply_them = mod.get_function("multiply_them")

        import numpy
        a = numpy.random.randn(400).astype(numpy.float32)
        b = numpy.random.randn(400).astype(numpy.float32)

        dest = numpy.zeros_like(a)
        multiply_them(
                drv.Out(dest), drv.In(a), drv.In(b),
                block=(400,1,1))
        assert la.norm(dest-a*b) == 0

    @mark_cuda_test
    def test_simple_kernel_2(self):
        mod = SourceModule("""
        __global__ void multiply_them(float *dest, float *a, float *b)
        {
          const int i = threadIdx.x;
          dest[i] = a[i] * b[i];
        }
        """)

        multiply_them = mod.get_function("multiply_them")

        import numpy
        a = numpy.random.randn(400).astype(numpy.float32)
        b = numpy.random.randn(400).astype(numpy.float32)
        a_gpu = drv.to_device(a)
        b_gpu = drv.to_device(b)

        dest = numpy.zeros_like(a)
        multiply_them(
                drv.Out(dest), a_gpu, b_gpu,
                block=(400,1,1))
        assert la.norm(dest-a*b) == 0

        # now try with offsets
        dest = numpy.zeros_like(a)
        multiply_them(
                drv.Out(dest), numpy.intp(a_gpu)+1, b_gpu,
                block=(399,1,1))

        assert la.norm(dest-a*b) == 0

    @mark_cuda_test
    def test_streamed_kernel(self):
        # this differs from the "simple_kernel" case in that *all* computation
        # and data copying is asynchronous. Observe how this necessitates the
        # use of page-locked memory.

        mod = SourceModule("""
        __global__ void multiply_them(float *dest, float *a, float *b)
        {
          const int i = threadIdx.x*blockDim.y + threadIdx.y;
          dest[i] = a[i] * b[i];
        }
        """)

        multiply_them = mod.get_function("multiply_them")

        import numpy
        shape = (32,8)
        a = drv.pagelocked_zeros(shape, dtype=numpy.float32)
        b = drv.pagelocked_zeros(shape, dtype=numpy.float32)
        a[:] = numpy.random.randn(*shape)
        b[:] = numpy.random.randn(*shape)

        a_gpu = drv.mem_alloc(a.nbytes)
        b_gpu = drv.mem_alloc(b.nbytes)

        strm = drv.Stream()
        drv.memcpy_htod_async(a_gpu, a, strm)
        drv.memcpy_htod_async(b_gpu, b, strm)
        strm.synchronize()

        dest = drv.pagelocked_empty_like(a)
        multiply_them(
                drv.Out(dest), a_gpu, b_gpu,
                block=shape+(1,), stream=strm)
        strm.synchronize()

        drv.memcpy_dtoh_async(a, a_gpu, strm)
        drv.memcpy_dtoh_async(b, b_gpu, strm)
        strm.synchronize()

        assert la.norm(dest-a*b) == 0

    @mark_cuda_test
    def test_gpuarray(self):
        import numpy
        a = numpy.arange(200000, dtype=numpy.float32)
        b = a + 17
        import pycuda.gpuarray as gpuarray
        a_g = gpuarray.to_gpu(a)
        b_g = gpuarray.to_gpu(b)
        diff = (a_g-3*b_g+(-a_g)).get() - (a-3*b+(-a))
        assert la.norm(diff) == 0

        diff = ((a_g*b_g).get()-a*b)
        assert la.norm(diff) == 0

    @mark_cuda_test
    def donottest_cublas_mixing():
        test_streamed_kernel()

        import pycuda.blas as blas

        shape = (10,)
        a = blas.ones(shape, dtype=numpy.float32)
        b = 33*blas.ones(shape, dtype=numpy.float32)
        assert ((-a+b).from_gpu() == 32).all()

        test_streamed_kernel()

    @mark_cuda_test
    def test_2d_texture(self):
        mod = SourceModule("""
        texture<float, 2, cudaReadModeElementType> mtx_tex;

        __global__ void copy_texture(float *dest)
        {
          int row = threadIdx.x;
          int col = threadIdx.y;
          int w = blockDim.y;
          dest[row*w+col] = tex2D(mtx_tex, row, col);
        }
        """)

        copy_texture = mod.get_function("copy_texture")
        mtx_tex = mod.get_texref("mtx_tex")

        shape = (3,4)
        a = numpy.random.randn(*shape).astype(numpy.float32)
        drv.matrix_to_texref(a, mtx_tex, order="F")

        dest = numpy.zeros(shape, dtype=numpy.float32)
        copy_texture(drv.Out(dest),
                block=shape+(1,),
                texrefs=[mtx_tex]
                )
        assert la.norm(dest-a) == 0

    @mark_cuda_test
    def test_multiple_2d_textures(self):
        mod = SourceModule("""
        texture<float, 2, cudaReadModeElementType> mtx_tex;
        texture<float, 2, cudaReadModeElementType> mtx2_tex;

        __global__ void copy_texture(float *dest)
        {
          int row = threadIdx.x;
          int col = threadIdx.y;
          int w = blockDim.y;
          dest[row*w+col] =
              tex2D(mtx_tex, row, col)
              +
              tex2D(mtx2_tex, row, col);
        }
        """)

        copy_texture = mod.get_function("copy_texture")
        mtx_tex = mod.get_texref("mtx_tex")
        mtx2_tex = mod.get_texref("mtx2_tex")

        shape = (3,4)
        a = numpy.random.randn(*shape).astype(numpy.float32)
        b = numpy.random.randn(*shape).astype(numpy.float32)
        drv.matrix_to_texref(a, mtx_tex, order="F")
        drv.matrix_to_texref(b, mtx2_tex, order="F")

        dest = numpy.zeros(shape, dtype=numpy.float32)
        copy_texture(drv.Out(dest),
                block=shape+(1,),
                texrefs=[mtx_tex, mtx2_tex]
                )
        assert la.norm(dest-a-b) < 1e-6

    @mark_cuda_test
    def test_multichannel_2d_texture(self):
        mod = SourceModule("""
        #define CHANNELS 4
        texture<float4, 2, cudaReadModeElementType> mtx_tex;

        __global__ void copy_texture(float *dest)
        {
          int row = threadIdx.x;
          int col = threadIdx.y;
          int w = blockDim.y;
          float4 texval = tex2D(mtx_tex, row, col);
          dest[(row*w+col)*CHANNELS + 0] = texval.x;
          dest[(row*w+col)*CHANNELS + 1] = texval.y;
          dest[(row*w+col)*CHANNELS + 2] = texval.z;
          dest[(row*w+col)*CHANNELS + 3] = texval.w;
        }
        """)

        copy_texture = mod.get_function("copy_texture")
        mtx_tex = mod.get_texref("mtx_tex")

        shape = (5,6)
        channels = 4
        a = numpy.asarray(
                numpy.random.randn(*((channels,)+shape)),
                dtype=numpy.float32, order="F")
        drv.bind_array_to_texref(
            drv.make_multichannel_2d_array(a, order="F"), mtx_tex)

        dest = numpy.zeros(shape+(channels,), dtype=numpy.float32)
        copy_texture(drv.Out(dest),
                block=shape+(1,),
                texrefs=[mtx_tex]
                )
        reshaped_a = a.transpose(1,2,0)
        #print reshaped_a
        #print dest
        assert la.norm(dest-reshaped_a) == 0

    @mark_cuda_test
    def test_multichannel_linear_texture(self):
        mod = SourceModule("""
        #define CHANNELS 4
        texture<float4, 1, cudaReadModeElementType> mtx_tex;

        __global__ void copy_texture(float *dest)
        {
          int i = threadIdx.x+blockDim.x*threadIdx.y;
          float4 texval = tex1Dfetch(mtx_tex, i);
          dest[i*CHANNELS + 0] = texval.x;
          dest[i*CHANNELS + 1] = texval.y;
          dest[i*CHANNELS + 2] = texval.z;
          dest[i*CHANNELS + 3] = texval.w;
        }
        """)

        copy_texture = mod.get_function("copy_texture")
        mtx_tex = mod.get_texref("mtx_tex")

        shape = (16, 16)
        channels = 4
        a = numpy.random.randn(*(shape+(channels,))).astype(numpy.float32)
        a_gpu = drv.to_device(a)
        mtx_tex.set_address(a_gpu, a.nbytes)
        mtx_tex.set_format(drv.array_format.FLOAT, 4)

        dest = numpy.zeros(shape+(channels,), dtype=numpy.float32)
        copy_texture(drv.Out(dest),
                block=shape+(1,),
                texrefs=[mtx_tex]
                )
        #print a
        #print dest
        assert la.norm(dest-a) == 0

    @mark_cuda_test
    def test_large_smem(self):
        n = 4000
        mod = SourceModule("""
        #include <stdio.h>

        __global__ void kernel(int *d_data)
        {
        __shared__ int sdata[%d];
        sdata[threadIdx.x] = threadIdx.x;
        d_data[threadIdx.x] = sdata[threadIdx.x];
        }
        """ % n)

        kernel = mod.get_function("kernel")

        import pycuda.gpuarray as gpuarray
        arg = gpuarray.zeros((n,), dtype=numpy.float32)

        kernel(arg, block=(1,1,1,), )

    @mark_cuda_test
    def test_bitlog(self):
        from pycuda.tools import bitlog2
        assert bitlog2(17) == 4
        assert bitlog2(0xaffe) == 15
        assert bitlog2(0x3affe) == 17
        assert bitlog2(0xcc3affe) == 27

    @mark_cuda_test
    def test_mempool_2(self):
        from pycuda.tools import DeviceMemoryPool as DMP
        from random import randrange

        for i in range(2000):
            s = randrange(1<<31) >> randrange(32)
            bin_nr = DMP.bin_number(s)
            asize = DMP.alloc_size(bin_nr)

            assert asize >= s, s
            assert DMP.bin_number(asize) == bin_nr, s
            assert asize < asize*(1+1/8)

    @mark_cuda_test
    def test_mempool(self):
        from pycuda.tools import bitlog2
        from pycuda.tools import DeviceMemoryPool

        pool = DeviceMemoryPool()
        maxlen = 10
        queue = []
        free, total = drv.mem_get_info()

        e0 = bitlog2(free)

        for e in range(e0-6, e0-4):
            for i in range(100):
                queue.append(pool.allocate(1<<e))
                if len(queue) > 10:
                    queue.pop(0)
        del queue
        pool.stop_holding()

    @mark_cuda_test
    def test_multi_context(self):
        if drv.get_version() < (2,0,0):
            return
        if drv.get_version() >= (2,2,0):
            if drv.Context.get_device().compute_mode == drv.compute_mode.EXCLUSIVE:
                return

        mem_a = drv.mem_alloc(50)
        ctx2 = drv.Context.get_device().make_context()
        mem_b = drv.mem_alloc(60)

        del mem_a
        del mem_b
        ctx2.detach()

    @mark_cuda_test
    def test_3d_texture(self):
        # adapted from code by Nicolas Pinto
        w = 2
        h = 4
        d = 8
        shape = (w, h, d)

        a = numpy.asarray(
                numpy.random.randn(*shape),
                dtype=numpy.float32, order="F")

        descr = drv.ArrayDescriptor3D()
        descr.width = w
        descr.height = h
        descr.depth = d
        descr.format = drv.dtype_to_array_format(a.dtype)
        descr.num_channels = 1
        descr.flags = 0

        ary = drv.Array(descr)

        copy = drv.Memcpy3D()
        copy.set_src_host(a)
        copy.set_dst_array(ary)
        copy.width_in_bytes = copy.src_pitch = a.strides[1]
        copy.src_height = copy.height = h
        copy.depth = d

        copy()

        mod = SourceModule("""
        texture<float, 3, cudaReadModeElementType> mtx_tex;

        __global__ void copy_texture(float *dest)
        {
          int x = threadIdx.x;
          int y = threadIdx.y;
          int z = threadIdx.z;
          int dx = blockDim.x;
          int dy = blockDim.y;
          int i = (z*dy + y)*dx + x;
          dest[i] = tex3D(mtx_tex, x, y, z);
          //dest[i] = x;
        }
        """)

        copy_texture = mod.get_function("copy_texture")
        mtx_tex = mod.get_texref("mtx_tex")

        mtx_tex.set_array(ary)

        dest = numpy.zeros(shape, dtype=numpy.float32, order="F")
        copy_texture(drv.Out(dest), block=shape, texrefs=[mtx_tex])
        assert la.norm(dest-a) == 0

    @mark_cuda_test
    def test_prepared_invocation(self):
        a = numpy.random.randn(4,4).astype(numpy.float32)
        a_gpu = drv.mem_alloc(a.size * a.dtype.itemsize)

        drv.memcpy_htod(a_gpu, a)

        mod = SourceModule("""
            __global__ void doublify(float *a)
            {
              int idx = threadIdx.x + threadIdx.y*blockDim.x;
              a[idx] *= 2;
            }
            """)

        func = mod.get_function("doublify")
        func.prepare("P", (4,4,1))
        func.prepared_call((1, 1), a_gpu)
        a_doubled = numpy.empty_like(a)
        drv.memcpy_dtoh(a_doubled, a_gpu)
        assert la.norm(a_doubled-2*a) == 0

        # now with offsets
        func.prepare("P", (15,1,1))
        a_quadrupled = numpy.empty_like(a)
        func.prepared_call((1, 1), int(a_gpu)+a.dtype.itemsize)
        drv.memcpy_dtoh(a_quadrupled, a_gpu)
        assert la.norm(a_quadrupled[1:]-4*a[1:]) == 0

    @mark_cuda_test
    def test_fp_textures(self):
        if drv.Context.get_device().compute_capability() < (1, 3):
            return

        for tp in [numpy.float32, numpy.float64]:
            from pycuda.tools import dtype_to_ctype

            tp_cstr = dtype_to_ctype(tp)
            mod = SourceModule("""
            #include <pycuda-helpers.hpp>

            texture<fp_tex_%(tp)s, 1, cudaReadModeElementType> my_tex;

            __global__ void copy_texture(%(tp)s *dest)
            {
              int i = threadIdx.x;
              dest[i] = fp_tex1Dfetch(my_tex, i);
            }
            """ % {"tp": tp_cstr})

            copy_texture = mod.get_function("copy_texture")
            my_tex = mod.get_texref("my_tex")

            import pycuda.gpuarray as gpuarray

            shape = (384,)
            a = numpy.random.randn(*shape).astype(tp)
            a_gpu = gpuarray.to_gpu(a)
            a_gpu.bind_to_texref_ext(my_tex, allow_double_hack=True)

            dest = numpy.zeros(shape, dtype=tp)
            copy_texture(drv.Out(dest),
                    block=shape+(1,1,),
                    texrefs=[my_tex])

            assert la.norm(dest-a) == 0

    @mark_cuda_test
    def test_constant_memory(self):
        # contributed by Andrew Wagner

        module = SourceModule("""
        __constant__ float const_array[32];

        __global__ void copy_constant_into_global(float* global_result_array)
        {
            global_result_array[threadIdx.x] = const_array[threadIdx.x];
        }
        """)

        copy_constant_into_global = module.get_function("copy_constant_into_global")
        const_array, _ = module.get_global('const_array')

        host_array = numpy.random.randint(0,255,(32,)).astype(numpy.float32)

        global_result_array = drv.mem_alloc_like(host_array)
        drv.memcpy_htod(const_array, host_array)

        copy_constant_into_global(
                global_result_array,  
                grid=(1, 1), block=(32, 1, 1))

        host_result_array = numpy.zeros_like(host_array)
        drv.memcpy_dtoh(host_result_array, global_result_array)

        assert (host_result_array == host_array).all



if __name__ == "__main__":
    # make sure that import failures get reported, instead of skipping the tests.
    import pycuda.autoinit

    import sys
    if len(sys.argv) > 1:
        exec sys.argv[1]
    else:
        from py.test.cmdline import main
        main([__file__])
