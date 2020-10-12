#!python 

import numpy
import pycuda.autoinit
import pycuda.driver as cuda

from pycuda.compiler import SourceModule

w = 7

mod = SourceModule("""
        #include<math.h>
        __global__ void diffusion(  int* result,int width, int height,float x,float y,float z) {

            int xIndex = blockDim.x * blockIdx.x + threadIdx.x;
            int yIndex = blockDim.y * blockIdx.y + threadIdx.y;

            int flatIndex = xIndex + width * yIndex;
            int topIndex = xIndex + width * (yIndex - 1);
            int bottomIndex = xIndex + width * (yIndex + 1);

            int inc = 1;

            result[flatIndex] = (result[flatIndex]-x)+(result[flatIndex]-y)+(result[flatIndex]-z);
        }

        """)

diff_func   = mod.get_function("diffusion")


def diffusion(res):

    x = numpy.float32(2)
    y = numpy.float32(1)
    z = numpy.float32(1)


    height, width = numpy.int32(len(res)), numpy.int32(len(res[0]))

    diff_func(
        cuda.InOut(res),
        width,
        height,x,y,z,
        block=(w,w,1)
        )

def run(res, step):

    diffusion(res)
    print(res)

res   = numpy.array([[0 \
                        for _ in range(0, w)]\
                        for _ in range(0, w)], dtype='int32')
print(res)
run(res, 0)

