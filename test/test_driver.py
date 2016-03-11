from __future__ import division
from __future__ import absolute_import
from __future__ import print_function
import numpy as np
import numpy.linalg as la
from pycuda.tools import mark_cuda_test, dtype_to_ctype
import pytest
from six.moves import range


def have_pycuda():
    try:
        import pycuda  # noqa
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
        z = np.random.randn(400).astype(np.float32)
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

        a = np.random.randn(400).astype(np.float32)
        b = np.random.randn(400).astype(np.float32)

        dest = np.zeros_like(a)
        multiply_them(
                drv.Out(dest), drv.In(a), drv.In(b),
                block=(400, 1, 1))
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

        a = np.random.randn(400).astype(np.float32)
        b = np.random.randn(400).astype(np.float32)
        a_gpu = drv.to_device(a)
        b_gpu = drv.to_device(b)

        dest = np.zeros_like(a)
        multiply_them(
                drv.Out(dest), a_gpu, b_gpu,
                block=(400, 1, 1))
        assert la.norm(dest-a*b) == 0

        drv.Context.synchronize()
        # now try with offsets
        dest = np.zeros_like(a)
        multiply_them(
                drv.Out(dest), np.intp(a_gpu)+a.itemsize, b_gpu,
                block=(399, 1, 1))

        assert la.norm((dest[:-1]-a[1:]*b[:-1])) == 0

    @mark_cuda_test
    def test_vector_types(self):
        mod = SourceModule("""
        __global__ void set_them(float3 *dest, float3 x)
        {
          const int i = threadIdx.x;
          dest[i] = x;
        }
        """)

        set_them = mod.get_function("set_them")
        a = gpuarray.vec.make_float3(1, 2, 3)
        dest = np.empty((400), gpuarray.vec.float3)

        set_them(drv.Out(dest), a, block=(400,1,1))
        assert (dest == a).all()

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

        shape = (32, 8)
        a = drv.pagelocked_zeros(shape, dtype=np.float32)
        b = drv.pagelocked_zeros(shape, dtype=np.float32)
        a[:] = np.random.randn(*shape)
        b[:] = np.random.randn(*shape)

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
        a = np.arange(200000, dtype=np.float32)
        b = a + 17
        import pycuda.gpuarray as gpuarray
        a_g = gpuarray.to_gpu(a)
        b_g = gpuarray.to_gpu(b)
        diff = (a_g-3*b_g+(-a_g)).get() - (a-3*b+(-a))
        assert la.norm(diff) == 0

        diff = ((a_g*b_g).get()-a*b)
        assert la.norm(diff) == 0

    @mark_cuda_test
    def donottest_cublas_mixing(self):
        self.test_streamed_kernel()

        import pycuda.blas as blas

        shape = (10,)
        a = blas.ones(shape, dtype=np.float32)
        b = 33*blas.ones(shape, dtype=np.float32)
        assert ((-a+b).from_gpu() == 32).all()

        self.test_streamed_kernel()

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

        shape = (3, 4)
        a = np.random.randn(*shape).astype(np.float32)
        drv.matrix_to_texref(a, mtx_tex, order="F")

        dest = np.zeros(shape, dtype=np.float32)
        copy_texture(
                drv.Out(dest),
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
        a = np.random.randn(*shape).astype(np.float32)
        b = np.random.randn(*shape).astype(np.float32)
        drv.matrix_to_texref(a, mtx_tex, order="F")
        drv.matrix_to_texref(b, mtx2_tex, order="F")

        dest = np.zeros(shape, dtype=np.float32)
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

        shape = (5, 6)
        channels = 4
        a = np.asarray(
                np.random.randn(*((channels,)+shape)),
                dtype=np.float32, order="F")
        drv.bind_array_to_texref(
            drv.make_multichannel_2d_array(a, order="F"), mtx_tex)

        dest = np.zeros(shape+(channels,), dtype=np.float32)
        copy_texture(
                drv.Out(dest),
                block=shape+(1,),
                texrefs=[mtx_tex]
                )
        reshaped_a = a.transpose(1, 2, 0)
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
        a = np.random.randn(*(shape+(channels,))).astype(np.float32)
        a_gpu = drv.to_device(a)
        mtx_tex.set_address(a_gpu, a.nbytes)
        mtx_tex.set_format(drv.array_format.FLOAT, 4)

        dest = np.zeros(shape+(channels,), dtype=np.float32)
        copy_texture(drv.Out(dest),
                block=shape+(1,),
                texrefs=[mtx_tex]
                )
        #print a
        #print dest
        assert la.norm(dest-a) == 0

    @mark_cuda_test
    def test_2d_fp_textures(self):
        orden = "F"
        npoints = 32

        for prec in [np.int16,np.float32,np.float64,np.complex64,np.complex128]:
            prec_str = dtype_to_ctype(prec)
            if prec == np.complex64: fpName_str = 'fp_tex_cfloat'
            elif prec == np.complex128: fpName_str = 'fp_tex_cdouble'
            elif prec == np.float64: fpName_str = 'fp_tex_double'
            else: fpName_str = prec_str
            A_cpu = np.zeros([npoints,npoints],order=orden,dtype=prec)
            A_cpu[:] = np.random.rand(npoints,npoints)[:]
            A_gpu = gpuarray.zeros(A_cpu.shape,dtype=prec,order=orden)

            myKern = '''
            #include <pycuda-helpers.hpp>
            texture<fpName, 2, cudaReadModeElementType> mtx_tex;

            __global__ void copy_texture(cuPres *dest)
            {
              int row = blockIdx.x*blockDim.x + threadIdx.x;
              int col = blockIdx.y*blockDim.y + threadIdx.y;

              dest[row + col*blockDim.x*gridDim.x] = fp_tex2D(mtx_tex, col, row);
            }
            '''
            myKern = myKern.replace('fpName',fpName_str)
            myKern = myKern.replace('cuPres',prec_str)
            mod = SourceModule(myKern)

            copy_texture = mod.get_function("copy_texture")
            mtx_tex = mod.get_texref("mtx_tex")
            cuBlock = (16,16,1)
            if cuBlock[0]>npoints:
                cuBlock = (npoints,npoints,1)
            cuGrid   = (npoints//cuBlock[0]+1*(npoints % cuBlock[0] != 0 ),npoints//cuBlock[1]+1*(npoints % cuBlock[1] != 0 ),1)
            copy_texture.prepare('P',texrefs=[mtx_tex])
            cudaArray = drv.np_to_array(A_cpu,orden,allowSurfaceBind=False)
            mtx_tex.set_array(cudaArray)
            copy_texture.prepared_call(cuGrid,cuBlock,A_gpu.gpudata)
            assert np.sum(np.abs(A_gpu.get()-np.transpose(A_cpu))) == np.array(0,dtype=prec)
            A_gpu.gpudata.free()

    @mark_cuda_test
    def test_2d_fp_texturesLayered(self):
        orden = "F"
        npoints = 32

        for prec in [np.int16,np.float32,np.float64,np.complex64,np.complex128]:
            prec_str = dtype_to_ctype(prec)
            if prec == np.complex64: fpName_str = 'fp_tex_cfloat'
            elif prec == np.complex128: fpName_str = 'fp_tex_cdouble'
            elif prec == np.float64: fpName_str = 'fp_tex_double'
            else: fpName_str = prec_str
            A_cpu = np.zeros([npoints,npoints],order=orden,dtype=prec)
            A_cpu[:] = np.random.rand(npoints,npoints)[:]
            A_gpu = gpuarray.zeros(A_cpu.shape,dtype=prec,order=orden)

            myKern = '''
            #include <pycuda-helpers.hpp>
            texture<fpName, cudaTextureType2DLayered, cudaReadModeElementType> mtx_tex;

            __global__ void copy_texture(cuPres *dest)
            {
              int row = blockIdx.x*blockDim.x + threadIdx.x;
              int col = blockIdx.y*blockDim.y + threadIdx.y;

              dest[row + col*blockDim.x*gridDim.x] = fp_tex2DLayered(mtx_tex, col, row, 1);
            }
            '''
            myKern = myKern.replace('fpName',fpName_str)
            myKern = myKern.replace('cuPres',prec_str)
            mod = SourceModule(myKern)

            copy_texture = mod.get_function("copy_texture")
            mtx_tex = mod.get_texref("mtx_tex")
            cuBlock = (16,16,1)
            if cuBlock[0]>npoints:
                cuBlock = (npoints,npoints,1)
            cuGrid   = (npoints//cuBlock[0]+1*(npoints % cuBlock[0] != 0 ),npoints//cuBlock[1]+1*(npoints % cuBlock[1] != 0 ),1)
            copy_texture.prepare('P',texrefs=[mtx_tex])
            cudaArray = drv.np_to_array(A_cpu,orden,allowSurfaceBind=True)
            mtx_tex.set_array(cudaArray)
            copy_texture.prepared_call(cuGrid,cuBlock,A_gpu.gpudata)
            assert np.sum(np.abs(A_gpu.get()-np.transpose(A_cpu))) == np.array(0,dtype=prec)
            A_gpu.gpudata.free()

    @mark_cuda_test
    def test_3d_fp_textures(self):
        orden = "C"
        npoints = 32

        for prec in [np.int16,np.float32,np.float64,np.complex64,np.complex128]:
            prec_str = dtype_to_ctype(prec)
            if prec == np.complex64: fpName_str = 'fp_tex_cfloat'
            elif prec == np.complex128: fpName_str = 'fp_tex_cdouble'
            elif prec == np.float64: fpName_str = 'fp_tex_double'
            else: fpName_str = prec_str
            A_cpu = np.zeros([npoints,npoints,npoints],order=orden,dtype=prec)
            A_cpu[:] = np.random.rand(npoints,npoints,npoints)[:]
            A_gpu = gpuarray.zeros(A_cpu.shape,dtype=prec,order=orden)

            myKern = '''
            #include <pycuda-helpers.hpp>
            texture<fpName, 3, cudaReadModeElementType> mtx_tex;

            __global__ void copy_texture(cuPres *dest)
            {
              int row   = blockIdx.x*blockDim.x + threadIdx.x;
              int col   = blockIdx.y*blockDim.y + threadIdx.y;
              int slice = blockIdx.z*blockDim.z + threadIdx.z;
              dest[row + col*blockDim.x*gridDim.x + slice*blockDim.x*gridDim.x*blockDim.y*gridDim.y] = fp_tex3D(mtx_tex, slice, col, row);
            }
            '''
            myKern = myKern.replace('fpName',fpName_str)
            myKern = myKern.replace('cuPres',prec_str)
            mod = SourceModule(myKern)

            copy_texture = mod.get_function("copy_texture")
            mtx_tex = mod.get_texref("mtx_tex")
            cuBlock = (8,8,8)
            if cuBlock[0]>npoints:
                cuBlock = (npoints,npoints,npoints)
            cuGrid   = (npoints//cuBlock[0]+1*(npoints % cuBlock[0] != 0 ),npoints//cuBlock[1]+1*(npoints % cuBlock[1] != 0 ),npoints//cuBlock[2]+1*(npoints % cuBlock[1] != 0 ))
            copy_texture.prepare('P',texrefs=[mtx_tex])
            cudaArray = drv.np_to_array(A_cpu,orden,allowSurfaceBind=False)
            mtx_tex.set_array(cudaArray)
            copy_texture.prepared_call(cuGrid,cuBlock,A_gpu.gpudata)
            assert np.sum(np.abs(A_gpu.get()-np.transpose(A_cpu))) == np.array(0,dtype=prec)
            A_gpu.gpudata.free()

    @mark_cuda_test
    def test_3d_fp_surfaces(self):
        orden = "C"
        npoints = 32

        for prec in [np.int16,np.float32,np.float64,np.complex64,np.complex128]:
            prec_str = dtype_to_ctype(prec)
            if prec == np.complex64:
                fpName_str = 'fp_tex_cfloat'
                A_cpu = np.zeros([npoints,npoints,npoints],order=orden,dtype=prec)
                A_cpu[:].real = np.random.rand(npoints,npoints,npoints)[:]
                A_cpu[:].imag = np.random.rand(npoints,npoints,npoints)[:]
            elif prec == np.complex128:
                fpName_str = 'fp_tex_cdouble'
                A_cpu = np.zeros([npoints,npoints,npoints],order=orden,dtype=prec)
                A_cpu[:].real = np.random.rand(npoints,npoints,npoints)[:]
                A_cpu[:].imag = np.random.rand(npoints,npoints,npoints)[:]
            elif prec == np.float64:
                fpName_str = 'fp_tex_double'
                A_cpu = np.zeros([npoints,npoints,npoints],order=orden,dtype=prec)
                A_cpu[:] = np.random.rand(npoints,npoints,npoints)[:]
            else:
                fpName_str = prec_str
                A_cpu = np.zeros([npoints,npoints,npoints],order=orden,dtype=prec)
                A_cpu[:] = np.random.rand(npoints,npoints,npoints)[:]*100.

            A_gpu = gpuarray.to_gpu(A_cpu) # Array randomized

            myKernRW = '''
            #include <pycuda-helpers.hpp>

            surface<void, cudaSurfaceType3D> mtx_tex;

            __global__ void copy_texture(cuPres *dest, int rw)
            {
              int row   = blockIdx.x*blockDim.x + threadIdx.x;
              int col   = blockIdx.y*blockDim.y + threadIdx.y;
              int slice = blockIdx.z*blockDim.z + threadIdx.z;
              int tid = row + col*blockDim.x*gridDim.x + slice*blockDim.x*gridDim.x*blockDim.y*gridDim.y;
              if (rw==0){
              cuPres aux = dest[tid];
              fp_surf3Dwrite(aux, mtx_tex, row, col, slice,cudaBoundaryModeClamp);}
              else {
              cuPres aux = 0;
              fp_surf3Dread(&aux, mtx_tex, slice, col, row, cudaBoundaryModeClamp);
              dest[tid] = aux;
              }
            }
            '''
            myKernRW = myKernRW.replace('fpName',fpName_str)
            myKernRW = myKernRW.replace('cuPres',prec_str)
            modW = SourceModule(myKernRW)

            copy_texture = modW.get_function("copy_texture")
            mtx_tex = modW.get_surfref("mtx_tex")
            cuBlock = (8,8,8)
            if cuBlock[0]>npoints:
                cuBlock = (npoints,npoints,npoints)
            cuGrid   = (npoints//cuBlock[0]+1*(npoints % cuBlock[0] != 0 ),npoints//cuBlock[1]+1*(npoints % cuBlock[1] != 0 ),npoints//cuBlock[2]+1*(npoints % cuBlock[1] != 0 ))
            copy_texture.prepare('Pi')#,texrefs=[mtx_tex])
            A_gpu2 = gpuarray.zeros_like(A_gpu) # To initialize surface with zeros
            cudaArray = drv.gpuarray_to_array(A_gpu2,orden,allowSurfaceBind=True)
            A_cpu = A_gpu.get() # To remember original array
            mtx_tex.set_array(cudaArray)
            copy_texture.prepared_call(cuGrid,cuBlock,A_gpu.gpudata, np.int32(0)) # Write random array
            copy_texture.prepared_call(cuGrid,cuBlock,A_gpu.gpudata, np.int32(1)) # Read, but transposed
            assert np.sum(np.abs(A_gpu.get()-np.transpose(A_cpu))) == np.array(0,dtype=prec)
            A_gpu.gpudata.free()

    @mark_cuda_test
    def test_2d_fp_surfaces(self):
        orden = "C"
        npoints = 32

        for prec in [np.int16,np.float32,np.float64,np.complex64,np.complex128]:
            prec_str = dtype_to_ctype(prec)
            if prec == np.complex64: fpName_str = 'fp_tex_cfloat'
            elif prec == np.complex128: fpName_str = 'fp_tex_cdouble'
            elif prec == np.float64: fpName_str = 'fp_tex_double'
            else: fpName_str = prec_str
            A_cpu = np.zeros([npoints,npoints],order=orden,dtype=prec)
            A_cpu[:] = np.random.rand(npoints,npoints)[:]
            A_gpu = gpuarray.to_gpu(A_cpu) # Array randomized

            myKernRW = '''
            #include <pycuda-helpers.hpp>

            surface<void, cudaSurfaceType2DLayered> mtx_tex;

            __global__ void copy_texture(cuPres *dest, int rw)
            {
              int row   = blockIdx.x*blockDim.x + threadIdx.x;
              int col   = blockIdx.y*blockDim.y + threadIdx.y;
              int layer = 1;
              int tid = row + col*blockDim.x*gridDim.x ;
              if (rw==0){
              cuPres aux = dest[tid];
              fp_surf2DLayeredwrite(aux, mtx_tex, row, col, layer,cudaBoundaryModeClamp);}
              else {
              cuPres aux = 0;
              fp_surf2DLayeredread(&aux, mtx_tex, col, row, layer, cudaBoundaryModeClamp);
              dest[tid] = aux;
              }
            }
            '''
            myKernRW = myKernRW.replace('fpName',fpName_str)
            myKernRW = myKernRW.replace('cuPres',prec_str)
            modW = SourceModule(myKernRW)

            copy_texture = modW.get_function("copy_texture")
            mtx_tex = modW.get_surfref("mtx_tex")
            cuBlock = (8,8,1)
            if cuBlock[0]>npoints:
                cuBlock = (npoints,npoints,1)
            cuGrid   = (npoints//cuBlock[0]+1*(npoints % cuBlock[0] != 0 ),npoints//cuBlock[1]+1*(npoints % cuBlock[1] != 0 ),1)
            copy_texture.prepare('Pi')#,texrefs=[mtx_tex])
            A_gpu2 = gpuarray.zeros_like(A_gpu) # To initialize surface with zeros
            cudaArray = drv.gpuarray_to_array(A_gpu2,orden,allowSurfaceBind=True)
            A_cpu = A_gpu.get() # To remember original array
            mtx_tex.set_array(cudaArray)
            copy_texture.prepared_call(cuGrid,cuBlock,A_gpu.gpudata, np.int32(0)) # Write random array
            copy_texture.prepared_call(cuGrid,cuBlock,A_gpu.gpudata, np.int32(1)) # Read, but transposed
            assert np.sum(np.abs(A_gpu.get()-np.transpose(A_cpu))) == np.array(0,dtype=prec)
            A_gpu.gpudata.free()

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
        arg = gpuarray.zeros((n,), dtype=np.float32)

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

        a = np.asarray(
                np.random.randn(*shape),
                dtype=np.float32, order="F")

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

        dest = np.zeros(shape, dtype=np.float32, order="F")
        copy_texture(drv.Out(dest), block=shape, texrefs=[mtx_tex])
        assert la.norm(dest-a) == 0

    @mark_cuda_test
    def test_prepared_invocation(self):
        a = np.random.randn(4,4).astype(np.float32)
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
        func.prepare("P")
        func.prepared_call((1, 1), (4,4,1), a_gpu, shared_size=20)
        a_doubled = np.empty_like(a)
        drv.memcpy_dtoh(a_doubled, a_gpu)
        print (a)
        print (a_doubled)
        assert la.norm(a_doubled-2*a) == 0

        # now with offsets
        func.prepare("P")
        a_quadrupled = np.empty_like(a)
        func.prepared_call((1, 1), (15,1,1), int(a_gpu)+a.dtype.itemsize)
        drv.memcpy_dtoh(a_quadrupled, a_gpu)
        assert la.norm(a_quadrupled[1:]-4*a[1:]) == 0

    @mark_cuda_test
    def test_prepared_with_vector(self):
        cuda_source = r'''
        __global__ void cuda_function(float3 input)
        {
        float3 result = make_float3(input.x, input.y, input.z);
        }
        '''

        mod = SourceModule(cuda_source, cache_dir=False, keep=False)

        kernel = mod.get_function("cuda_function")
        arg_types = [gpuarray.vec.float3]

        kernel.prepare(arg_types)
        kernel.prepared_call((1, 1, 1), (1, 1, 1),
                gpuarray.vec.make_float3(0.0, 1.0, 2.0))

    @mark_cuda_test
    def test_fp_textures(self):
        if drv.Context.get_device().compute_capability() < (1, 3):
            return

        for tp in [np.float32, np.float64]:
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
            a = np.random.randn(*shape).astype(tp)
            a_gpu = gpuarray.to_gpu(a)
            a_gpu.bind_to_texref_ext(my_tex, allow_double_hack=True)

            dest = np.zeros(shape, dtype=tp)
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

        host_array = np.random.randint(0,255,(32,)).astype(np.float32)

        global_result_array = drv.mem_alloc_like(host_array)
        drv.memcpy_htod(const_array, host_array)

        copy_constant_into_global(
                global_result_array,
                grid=(1, 1), block=(32, 1, 1))

        host_result_array = np.zeros_like(host_array)
        drv.memcpy_dtoh(host_result_array, global_result_array)

        assert (host_result_array == host_array).all

    @mark_cuda_test
    def test_register_host_memory(self):
        if drv.get_version() < (4,):
            from py.test import skip
            skip("register_host_memory only exists on CUDA 4.0 and later")

        import sys
        if sys.platform == "darwin":
            from py.test import skip
            skip("register_host_memory is not supported on OS X")

        import resource

        a = drv.aligned_empty((2**20,), np.float64,
            alignment=resource.getpagesize())
        a_pin = drv.register_host_memory(a)

        gpu_ary = drv.mem_alloc_like(a)
        stream = drv.Stream()
        drv.memcpy_htod_async(gpu_ary, a_pin, stream)
        drv.Context.synchronize()

    @pytest.mark.xfail
    @mark_cuda_test
    # https://github.com/inducer/pycuda/issues/45
    def test_recursive_launch(self):
        # Test contributed by Aditya Avinash Atluri

        if drv.Context.get_device().compute_capability() < (3, 5):
            from pytest import skip
            skip("need compute capability 3.5 or higher for dynamic parallelism")

        cuda_string = """
        __device__ void  saxpy(double* s, float a, long* p, int b, long* q)
        {
            int tx = threadIdx.x;
            s[tx] = a*p[tx]+b*q[tx];
        }

        __global__ void  sub(long* p, long* q, long* d)
        {
            int tx = threadIdx.x;
            p[tx] = 2*p[tx];
            d[tx] = p[tx]-q[tx];
        }

        __device__ long add(long p, long q)
        {
            p = p+1;
            return p+q;
        }

        __global__ void math(long* a, long* b, long* c, long* d, long* e, double* f)
        {
            int tx = threadIdx.x;
            __shared__ long x[100];
            x[tx] = a[tx + 0];
            __shared__ long y[100];
            y[tx] = b[tx + 0];
            c[tx]=add(x[tx],y[tx]);
            dim3 dimGrid_sub(1,1,1);
            dim3 dimBlock_sub(100,1,1);
            sub<<<dimGrid_sub,dimBlock_sub>>>(a,b,d);
            saxpy(f,1.0345,x,-2,y);
        }
        """

        def math(a, b, c, d, e, f):
            a_gpu = drv.mem_alloc(a.nbytes)
            b_gpu = drv.mem_alloc(b.nbytes)
            c_gpu = drv.mem_alloc(c.nbytes)
            d_gpu = drv.mem_alloc(d.nbytes)
            e_gpu = drv.mem_alloc(e.nbytes)
            f_gpu = drv.mem_alloc(f.nbytes)

            drv.memcpy_htod(a_gpu, a)
            drv.memcpy_htod(b_gpu, b)

            mod = SourceModule(cuda_string,
                    options=['-rdc=true', '-lcudadevrt'],
                    keep=True)

            func = mod.get_function("math")
            func(a_gpu, b_gpu, c_gpu, d_gpu, e_gpu, f_gpu,
                    block=(100, 1, 1), grid=(1, 1, 1))

            drv.memcpy_dtoh(c, c_gpu)
            drv.memcpy_dtoh(d, d_gpu)
            drv.memcpy_dtoh(e, e_gpu)
            drv.memcpy_dtoh(f, f_gpu)

            #print(c,d,e,f)

        a = np.random.randint(10, size=100)
        b = np.random.randint(10, size=100)
        c = np.empty_like(a)
        d = np.empty_like(a)
        e = np.empty_like(a)
        f = np.array(a, dtype='d')

        math(a, b, c, d, e, f)


def test_import_pyopencl_before_pycuda():
    try:
        import pyopencl  # noqa
    except ImportError:
        return
    import pycuda.driver  # noqa


if __name__ == "__main__":
    # make sure that import failures get reported, instead of skipping the tests.
    import pycuda.autoinit  # noqa

    import sys
    if len(sys.argv) > 1:
        exec (sys.argv[1])
    else:
        from py.test.cmdline import main
        main([__file__])
