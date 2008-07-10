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


    def test_index_access(self):
        """tests the index based access"""
        a = numpy.array([1,2,3,4,5]).astype(numpy.float32)
        a_cpu = self.create_array(a)

        for i in range(0,5):
            self.assert_(a[i] == a_cpu[i])


    def test_iterate(self):
        """tests the iteration methods"""
        a = numpy.array([1,2,3,4,5]).astype(numpy.float32)
        a_cpu = self.create_array(a)

        for i in a_cpu:
            #ok iteration works
            self.assert_(True)


    def test_pow_array(self):
        """tests the pow function based on arrays"""

        a = numpy.array([1,2,3,4,5]).astype(numpy.float32)
        a_cpu = self.create_array(a)

        result = pow(a_cpu,a_cpu).get()

        for i in range(0,5):
            self.assert_(abs(pow(a[i],a[i]) - result[i]) < 1e-3)

        result = a_cpu**a_cpu

        for i in range(0,5):
            self.assert_(abs(pow(a[i],a[i]) - result[i]) < 1e-3)


    def test_pow_number(self):
        """tests the pow function based on a number"""

        a = numpy.array([1,2,3,4,5,6,7,8,9,10]).astype(numpy.float32)
        a_cpu = self.create_array(a)
 
        result = pow(a_cpu,2).get()

        for i in range(0,10):
            self.assert_(abs(pow(a[i],2) - result[i]) < 1e-3)
       

    def test_abs(self):
        """test if the abs function works"""
        a = gpuarray.arrange(111)
        a = a * -1

        res = a.get()

        for i in range (111):
            self.assert_(res[i] <= 0)

        a = abs(a)

        res = a.get()

        for i in range (111):
            self.assert_(res[i] >= 0)
            self.assert_(res[i] == i)


        for i in range(100,200):
            a = gpuarray.arrange(500 * i)
            self.assert_(a[len(a)-1] == len(a)-1)

    def test_arrange(self):
        """test the arrangement of the array"""
        a = gpuarray.arrange(12)

        res = a.get()

        for i in range(12):
            self.assert_(res[i] ==i)

    def test_reverse(self):
        """test if the reverse works"""
        a = numpy.array([1,2,3,4,5,6,7,8,9,10]).astype(numpy.float32)
        a_cpu = self.create_array(a)

        a_cpu = a_cpu.reverse()


        b = a_cpu.get()

        for i in range(0,10):
            self.assert_(a[len(a)-1-i] == b[i])
        

    def test_is_matrix(self):
        """tests if the gpu array is a matrix"""
        a = numpy.array([1,2,3,4,5,6,7,8,9,10]).astype(numpy.float32)
        a.shape = 2,5

        b = numpy.array([1,2,3,4,5,6,7,8,9,10]).astype(numpy.float32)
        
        self.assert_(self.create_array(a).is_matrix() == True)
        self.assert_(self.create_array(b).is_matrix() == False)


    def test_len(self):
        """test the len"""
        a = numpy.array([1,2,3,4,5,6,7,8,9,10]).astype(numpy.float32)
        a_cpu = self.create_array(a)
        self.assert_(len(a_cpu) == 10)


    def test_shape(self):
        a = numpy.array([1,2,3,4,5,6,7,8,9,10]).astype(numpy.float32)
        a_cpu = self.create_array(a)
        self.assert_(len(a_cpu._shape()) == 1)


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


        #test with a large array
        a = numpy.arange(50000).astype(numpy.float32)
        
        a_gpu = self.create_array(a)

        a_doubled = (2 * a_gpu).get()

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

