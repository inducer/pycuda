import pycuda.simplearray as simplearray
import numpy
import unittest
import test_abstract_array as test

class TestSimpleArray(test.TestAbstractArray):
    """tests a simple array"""


    def create_array(self,array):
        """creates an array an can be overwritten to replace the implementation"""
        return simplearray.to_gpu(array)

    def test_matrix(self):
        #initialize data with 0 of a size 100x100
        a = simplearray.matrix(100,100)
        
        #initialize data with 20 of a size 100x100
        b = simplearray.matrix(100,100,20)
        
        #initialize data with 30 of a size 100x100
        c = simplearray.matrix(100,100,30)
        
        #simple formula to run over all the data
        d = (a - 5) + (b + 2) / c
        
        for i in a:
            print i
        
if __name__ == '__main__':
    unittest.main()
