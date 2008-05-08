


def main():
    import pycuda.blas as blas
    import pyublas

    import numpy
    shape = (1000000,)
    a = numpy.ones(shape, dtype=numpy.float32)
    b = 33*numpy.ones(shape, dtype=numpy.float32)
    result = numpy.empty(shape, dtype=numpy.float32)

    a_gpu = blas.DevicePtrFloat32(shape[0])
    a_gpu.set(a)
    b_gpu = blas.DevicePtrFloat32(shape[0])
    b_gpu.set(b)

    blas.axpy(shape[0], 1, a_gpu, 1, b_gpu, 1)

    b_gpu.get(result)
    print result[:10], result[-10:]







if __name__ == "__main__":
    main()
