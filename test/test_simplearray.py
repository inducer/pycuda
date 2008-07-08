import pycuda.simplearray as simplearray
import numpy
import unittest
import test_abstract_array as test

class TestSimpleArray(test.TestAbstractArray):
    """tests a simple array"""


    def create_array(self,array):
        """creates an array an can be overwritten to replace the implementation"""
        return simplearray.to_gpu(array)

if __name__ == '__main__':
    unittest.main()
