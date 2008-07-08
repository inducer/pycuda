#! /usr/bin/env python
import pycuda.driver as drv
import numpy
import numpy.linalg as la
import unittest
import pycuda.gpuarray as gpuarray
import sys

class TestAbstractArray(unittest.TestCase):                                      
    """tests the array classes"""

    def create_array(self,array):
        """is supposed to create an array and needs to be overwritten"""
        return None

    def test_multiply(self):
        """Tests the muliplication of an array with a scalar. """

        #test data
        a = numpy.array([1,2,3,4,5,6,7,8,9,10]).astype(numpy.float32)

        #convert a to a gpu object
        a_gpu = self.create_array(a)

        #multipy all elements in a_gpu * 2, this should run on the gpu
        a_doubled = (2*a_gpu).get()

        #check that the result is like we expsect it
        for i in range(0,a.size):
           self.assertEqual(a[i] * 2,a_doubled[i])


    def test_multiply_array(self):
        """Tests the multiplaction of two arrays."""

        #test data
        a = numpy.array([1,2,3,4,5,6,7,8,9,10]).astype(numpy.float32)

        #convert a to a gpu object
        a_gpu = self.create_array(a)
        b_gpu = self.create_array(a)

        #multipy all elements in a_gpu * b_gpu, this should run on the gpu
        a_doubled = (b_gpu*a_gpu).get()

        #check that the result is like we expsect it
        for i in range(0,a.size):
           self.assertEqual(a[i] * a[i],a_doubled[i])


    def test_addition_array(self):
        """Tests the addition of two arrays."""

        #test data
        a = numpy.array([1,2,3,4,5,6,7,8,9,10]).astype(numpy.float32)

        #convert a to a gpu object
        a_gpu = self.create_array(a)
        b_gpu = self.create_array(a)

        #add all elements in a_gpu with b_gpu, this should run on the gpu
        a_added = (b_gpu+a_gpu).get()

        #check that the result is like we expsect it
        for i in range(0,a.size):
           self.assertEqual(a[i] + a[i],a_added[i])


    def test_addition_scalar(self):
        """Tests the addition of an array and a scalar."""

        #test data
        a = numpy.array([1,2,3,4,5,6,7,8,9,10]).astype(numpy.float32)

        #convert a to a gpu object
        a_gpu = self.create_array(a)

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
        """Tests the substraction of two arrays."""
        #test data
        a = numpy.array([1,2,3,4,5,6,7,8,9,10]).astype(numpy.float32)
        b = numpy.array([10,20,30,40,50,60,70,80,90,100]).astype(numpy.float32)

        #convert a to a gpu object
        a_gpu = self.create_array(a)
        b_gpu = self.create_array(b)

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
        """Tests the substraction of an array and a scalar."""

        #test data
        a = numpy.array([1,2,3,4,5,6,7,8,9,10]).astype(numpy.float32)

        #convert a to a gpu object
        a_gpu = self.create_array(a)

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
        """Tests the division of an array and a scalar."""

        #test data
        a = numpy.array([1,2,3,4,5,6,7,8,9,10]).astype(numpy.float32)

        #convert a to a gpu object
        a_gpu = self.create_array(a)

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
        """Tests the division of an array and a scalar. """

        #test data
        a = numpy.array([10,20,30,40,50,60,70,80,90,100]).astype(numpy.float32)
        b = numpy.array([10,10,10,10,10,10,10,10,10,10]).astype(numpy.float32)

        #convert a to a gpu object
        a_gpu = self.create_array(a)
        b_gpu = self.create_array(b)

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

