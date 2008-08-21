#! /usr/bin/env python
import pycuda.autoinit
import pycuda.driver as drv
import numpy
import numpy.linalg as la
import unittest
import pycuda.gpuarray as gpuarray
import sys
import test_abstract_array

class TestGPUArray(test_abstract_array.TestAbstractArray):                                      
    """tests the gpu array class"""

    def create_array(self,array):
        """creates a gpu array"""
        return gpuarray.to_gpu(array)

if __name__ == '__main__':
    unittest.main()
