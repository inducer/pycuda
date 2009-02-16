import pycuda.autoinit
import pycuda.driver as drv
import pycuda.gpuarray as gpuarray
from pycuda.compiler import SourceModule
import numpy
import numpy.linalg as la




def test():
    expt = 20
    mod = SourceModule("""
    texture<float, 1, cudaReadModeElementType> ary_tex;

    __global__ void copy_texture(float *dest)
    {
      int i = threadIdx.x;
      while (i < (1<<%d))
      {
        dest[i] = tex1Dfetch(ary_tex, i);
        i += blockDim.x;
      }
    }
    """ % expt)

    copy_texture = mod.get_function("copy_texture")
    ary_tex = mod.get_texref("ary_tex")

    shape = (1<<expt,)
    a = numpy.random.randn(*shape).astype(numpy.float32)
    for i in range(0, shape[0], 2):
        a[i] = float('nan')

    a_gpu = gpuarray.to_gpu(a)
    a_gpu.bind_to_texref(ary_tex)

    dest = numpy.zeros_like(a)
    copy_texture(drv.Out(dest),
            block=(512,1,1,), 
            texrefs=[ary_tex]
            )

    for i in range(1, shape[0], 2):
        assert not numpy.isnan(dest[i])



if __name__ == "__main__":
    test()

