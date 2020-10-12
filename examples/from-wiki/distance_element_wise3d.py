#!python 
import pycuda.gpuarray as gpuarray
import pycuda.driver as cuda
import pycuda.tools as tools
import pycuda.autoinit
import numpy, random, time
from pycuda.curandom import rand as curand
from pycuda.elementwise import ElementwiseKernel as Elementwise

x = 50
y = 50
z = 2
width = 100
height = 100
depth = 100

def main():
    """
    Computes a set of distances from a given point in a search space in parallel on a GPU.
    """

    # Create an empty array to hold our points.
    n = gpuarray.zeros(shape=(x, y, z),
                    dtype=gpuarray.vec.float3)

    # Populate the array with randomized points from the search space.
    for k in range(z):
        for j in range(y):
            for i in range(x):
                n[i, j, k] = gpuarray.vec.make_float3(random.uniform(-width, width),
                                                    random.uniform(-height, height),
                                                    random.uniform(-depth, depth))

    # Declare our elementwise CUDA kernel.
    mod = Elementwise(
        arguments="float3 pt, float3 *ns, float *rs",
        operation="rs[i] = sqrt(pow(pt.x-ns[i].x,2)+pow(pt.y-ns[i].y,2)+pow(pt.z-ns[i].z,2))",
        name="euclidean_distance",
        preamble="#include <math.h>"
    )

    # Declare an empty results array.
    r = gpuarray.zeros(shape=(50, 50, 2), dtype=numpy.float32)
    start = cuda.Event()
    end = cuda.Event()
    start.record()
    # Call the kernel with a randomize point from the search space.
    mod(gpuarray.vec.make_float3(random.uniform(-width, width),
                                 random.uniform(-height, height),
                                 random.uniform(-width, width)), n, r)
    end.record()
    end.synchronize()
    print((start.time_till(end)))
    print(r)

if __name__ == '__main__':
    main()


