def main():
    import pycuda.blas as blas
    import pyublas

    import numpy
    shape = (10,)
    a = blas.ones(shape, dtype=numpy.float32)
    b = 33*blas.ones(shape, dtype=numpy.float32)

    print -a+b




if __name__ == "__main__":
    main()
