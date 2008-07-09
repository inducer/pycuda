#! /usr/bin/env python
import pycuda.driver as drv
import numpy
import numpy.linalg as la
import unittest
import pycuda.gpuarray as gpuarray
import sys
import test_abstract_array as test

#initialize the device and the driver
if not drv.was_context_created():
    drv.init()
    ctx = drv.Device(0).make_context()
    gpuarray.GPUArray.compile_kernels()

class TestGPUArray(test.TestAbstractArray):                                      
    """tests the gpu array class"""

    def create_array(self,array):
        """creates a gpu array"""
        return gpuarray.to_gpu(array)

if __name__ == '__main__':
    unittest.main()
