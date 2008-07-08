#! /usr/bin/env python
import pycuda.driver as drv
import numpy
import numpy.linalg as la
import unittest
import pycuda.gpuarray as gpuarray
import sys

"""tests the gpu array class"""
class TestGPUArrary(unittest.TestCase):                                      


    """
    
       Initillializing of the class and the cuda driver.  
    
    """
    def setUp(self):
        #initialize driver
        drv.init()


        #we should have a function to check if a context was already created
        try:
           ctx = drv.Device(0).make_context()
        except RuntimeError:
           #we are going to ignore the runtime error at this point
           #since its realted that we try to create more than one context
           ""

        #compile kernels
        gpuarray.GPUArray.compile_kernels()

        

    """
    
       Tests the muliplication of an array with a scalar.  
       
    """
    def test_multiply(self):

        #test data
        a = numpy.array([1,2,3,4,5,6,7,8,9,10]).astype(numpy.float32)

        #convert a to a gpu object
        a_gpu = gpuarray.to_gpu(a)

        #multipy all elements in a_gpu * 2, this should run on the gpu
        a_doubled = (2*a_gpu).get()

        #check that the result is like we expsect it
        for i in range(0,a.size):
           self.assertEqual(a[i] * 2,a_doubled[i])


    """
    
        Tests the multiplaction of two arrays.  
    
    """
    def test_multiply_array(self):

        #test data
        a = numpy.array([1,2,3,4,5,6,7,8,9,10]).astype(numpy.float32)

        #convert a to a gpu object
        a_gpu = gpuarray.to_gpu(a)
        b_gpu = gpuarray.to_gpu(a)

        #multipy all elements in a_gpu * b_gpu, this should run on the gpu
        a_doubled = (b_gpu*a_gpu).get()

        #check that the result is like we expsect it
        for i in range(0,a.size):
           self.assertEqual(a[i] * a[i],a_doubled[i])


    """
     
        Tests the addition of two arrays.  
     
    """
    def test_addition_array(self):

        #test data
        a = numpy.array([1,2,3,4,5,6,7,8,9,10]).astype(numpy.float32)

        #convert a to a gpu object
        a_gpu = gpuarray.to_gpu(a)
        b_gpu = gpuarray.to_gpu(a)

        #add all elements in a_gpu with b_gpu, this should run on the gpu
        a_added = (b_gpu+a_gpu).get()

        #check that the result is like we expsect it
        for i in range(0,a.size):
           self.assertEqual(a[i] + a[i],a_added[i])


    """
    
        Tests the addition of an array and a scalar.  
    
    """
    def test_addition_scalar(self):

        #test data
        a = numpy.array([1,2,3,4,5,6,7,8,9,10]).astype(numpy.float32)

        #convert a to a gpu object
        a_gpu = gpuarray.to_gpu(a)

        #add all elements in a_gpu to 7, this should run on the gpu
        a_added = (7+a_gpu).get()

        #check that the result is like we expsect it
        for i in range(0,a.size):
           self.assertEqual(7 + a[i],a_added[i])


        a_added = (a_gpu + 7).get()

        #check that the result is like we expsect it
        for i in range(0,a.size):
           self.assertEqual(7 + a[i],a_added[i])


    """
    
       Tests the substraction of two arrays.  
    
    """
    def test_substract_array(self):

        #test data
        a = numpy.array([1,2,3,4,5,6,7,8,9,10]).astype(numpy.float32)
        b = numpy.array([10,20,30,40,50,60,70,80,90,100]).astype(numpy.float32)

        #convert a to a gpu object
        a_gpu = gpuarray.to_gpu(a)
        b_gpu = gpuarray.to_gpu(b)

        #add all elements in a_gpu with b_gpu, this should run on the gpu
        a_substract = (a_gpu-b_gpu).get()

        #check that the result is like we expsect it
        for i in range(0,a.size):
           self.assertEqual(a[i] - b[i],a_substract[i])

        #add all elements in a_gpu with b_gpu, this should run on the gpu
        a_substract = (b_gpu-a_gpu).get()

        #check that the result is like we expsect it
        for i in range(0,a.size):
           self.assertEqual(b[i] - a[i],a_substract[i])



    """
    
        Tests the substraction of an array and a scalar.  
    
    """
    def test_substract_scalar(self):

        #test data
        a = numpy.array([1,2,3,4,5,6,7,8,9,10]).astype(numpy.float32)

        #convert a to a gpu object
        a_gpu = gpuarray.to_gpu(a)

        #substract from all elements 7 in a_gpu
        a_substract = (a_gpu-7).get()

        #check that the result is like we expsect it
        for i in range(0,a.size):
           self.assertEqual(a[i]-7,a_substract[i])

        #substract 7 from all elements in a_gpu
        a_substract = (7-a_gpu).get()

        #check that the result is like we expsect it
        for i in range(0,a.size):
           self.assertEqual(7-a[i],a_substract[i])




    """

        Tests the divition of an array and a scalar.

    """
    def test_divide_scalar(self):

        #test data
        a = numpy.array([1,2,3,4,5,6,7,8,9,10]).astype(numpy.float32)

        #convert a to a gpu object
        a_gpu = gpuarray.to_gpu(a)

        #divides the array by 2
        a_divide = (a_gpu/2).get()

        #check that the result is like we expsect it
        for i in range(0,a.size):
            self.assert_( abs(a[i]/2 - a_divide[i]) < 1e-3 )

        #divides the array by 2
        a_divide = (2/a_gpu).get()

        #check that the result is like we expsect it
        for i in range(0,a.size):
            self.assert_( abs(2/a[i] - a_divide[i]) < 1e-3 )

    """

        Tests the divition of an array and a scalar.

    """
    def test_divide_array(self):

        #test data
        a = numpy.array([10,20,30,40,50,60,70,80,90,100]).astype(numpy.float32)
        b = numpy.array([10,10,10,10,10,10,10,10,10,10]).astype(numpy.float32)

        #convert a to a gpu object
        a_gpu = gpuarray.to_gpu(a)
        b_gpu = gpuarray.to_gpu(b)

        #divides the array
        a_divide = (a_gpu/b_gpu).get()

        #check that the result is like we expsect it
        for i in range(0,a.size):
            self.assert_( abs(a[i]/b[i] - a_divide[i]) < 1e-3 )

        #divides the array
        a_divide = (b_gpu/a_gpu).get()

        #check that the result is like we expsect it
        for i in range(0,a.size):
           self.assert_( abs(b[i]/a[i] - a_divide[i]) < 1e-3 )


if __name__ == '__main__':
    unittest.main()

