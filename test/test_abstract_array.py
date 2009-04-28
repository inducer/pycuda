#! /usr/bin/env python
import pycuda.autoinit
import pycuda.driver as drv
import numpy
import numpy.linalg as la
import unittest
import pycuda.gpuarray as gpuarray
import sys

class TestAbstractArray(unittest.TestCase):                                      
    __test__ = False

    def make_test_array(self, array):
        """Turn a given numpy array into the kind of array that 
        is to be tested."""
        raise NotImplementedError

    def test_pow_array(self):
        a = numpy.array([1,2,3,4,5]).astype(numpy.float32)
        a_gpu = self.make_test_array(a)

        result = pow(a_gpu,a_gpu).get()

        for i in range(0,5):
            self.assert_(abs(pow(a[i],a[i]) - result[i]) < 1e-3)

        result = (a_gpu**a_gpu).get()

        for i in range(0,5):
            self.assert_(abs(pow(a[i],a[i]) - result[i]) < 1e-3)

    def test_pow_number(self):
        a = numpy.array([1,2,3,4,5,6,7,8,9,10]).astype(numpy.float32)
        a_cpu = self.make_test_array(a)
 
        result = pow(a_cpu,2).get()

        for i in range(0,10):
            self.assert_(abs(pow(a[i],2) - result[i]) < 1e-3)
       

    def test_abs(self):
        a = -gpuarray.arange(111, dtype=numpy.float32)
        res = a.get()

        for i in range(111):
            self.assert_(res[i] <= 0)

        a = abs(a)

        res = a.get()

        for i in range (111):
            self.assert_(abs(res[i]) >= 0)
            self.assert_(res[i] == i)


    def test_len(self):
        a = numpy.array([1,2,3,4,5,6,7,8,9,10]).astype(numpy.float32)
        a_cpu = self.make_test_array(a)
        self.assert_(len(a_cpu) == 10)

    def test_multiply(self):
        """Test the muliplication of an array with a scalar. """

        #test data
        a = numpy.array([1,2,3,4,5,6,7,8,9,10]).astype(numpy.float32)

        #convert a to a gpu object
        a_gpu = self.make_test_array(a)

        #multipy all elements in a_gpu * 2, this should run on the gpu
        a_doubled = (2*a_gpu).get()

        #check that the result is like we expsect it
        for i in range(0,a.size):
           self.assertEqual(a[i] * 2,a_doubled[i])


        #test with a large array
        a = numpy.arange(50000).astype(numpy.float32)
        
        a_gpu = self.make_test_array(a)

        a_doubled = (2 * a_gpu).get()

        #check that the result is like we expsect it
        for i in range(0,a.size):
           self.assertEqual(a[i] * 2,a_doubled[i])

        

    def test_multiply_array(self):
        """Test the multiplication of two arrays."""

        #test data
        a = numpy.array([1,2,3,4,5,6,7,8,9,10]).astype(numpy.float32)

        #convert a to a gpu object
        a_gpu = self.make_test_array(a)
        b_gpu = self.make_test_array(a)

        #multipy all elements in a_gpu * b_gpu, this should run on the gpu
        a_doubled = (b_gpu*a_gpu).get()

        #check that the result is like we expsect it
        for i in range(0,a.size):
           self.assertEqual(a[i] * a[i],a_doubled[i])


    def test_addition_array(self):
        """Test the addition of two arrays."""

        #test data
        a = numpy.array([1,2,3,4,5,6,7,8,9,10]).astype(numpy.float32)

        #convert a to a gpu object
        a_gpu = self.make_test_array(a)
        b_gpu = self.make_test_array(a)

        #add all elements in a_gpu with b_gpu, this should run on the gpu
        a_added = (b_gpu+a_gpu).get()

        #check that the result is like we expsect it
        for i in range(0,a.size):
           self.assertEqual(a[i] + a[i],a_added[i])


    def test_addition_scalar(self):
        """Test the addition of an array and a scalar."""

        #test data
        a = numpy.array([1,2,3,4,5,6,7,8,9,10]).astype(numpy.float32)

        #convert a to a gpu object
        a_gpu = self.make_test_array(a)

        #add all elements in a_gpu to 7, this should run on the gpu
        a_added = (7+a_gpu).get()

        #check that the result is like we expsect it
        for i in range(0,a.size):
           self.assertEqual(7 + a[i],a_added[i])


        a_added = (a_gpu + 7).get()

        #check that the result is like we expsect it
        for i in range(0,a.size):
           self.assertEqual(7 + a[i],a_added[i])


    def test_substract_array(self):
        """Test the substraction of two arrays."""
        #test data
        a = numpy.array([1,2,3,4,5,6,7,8,9,10]).astype(numpy.float32)
        b = numpy.array([10,20,30,40,50,60,70,80,90,100]).astype(numpy.float32)

        #convert a to a gpu object
        a_gpu = self.make_test_array(a)
        b_gpu = self.make_test_array(b)

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



    def test_substract_scalar(self):
        """Test the substraction of an array and a scalar."""

        #test data
        a = numpy.array([1,2,3,4,5,6,7,8,9,10]).astype(numpy.float32)

        #convert a to a gpu object
        a_gpu = self.make_test_array(a)

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




    def test_divide_scalar(self):
        """Test the division of an array and a scalar."""

        #test data
        a = numpy.array([1,2,3,4,5,6,7,8,9,10]).astype(numpy.float32)

        #convert a to a gpu object
        a_gpu = self.make_test_array(a)

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

    def test_divide_array(self):
        """Test the division of an array and a scalar. """

        #test data
        a = numpy.array([10,20,30,40,50,60,70,80,90,100]).astype(numpy.float32)
        b = numpy.array([10,10,10,10,10,10,10,10,10,10]).astype(numpy.float32)

        #convert a to a gpu object
        a_gpu = self.make_test_array(a)
        b_gpu = self.make_test_array(b)

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
