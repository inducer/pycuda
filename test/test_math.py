import pycuda.autoinit
import pycuda.gpuarray as gpuarray
import unittest
import pycuda.cumath as cumath
import math
import numpy

test_sample =  1<<13
    
print "sample size: ", test_sample

class TestMath(unittest.TestCase):
    def test_ceil(self):
        """tests if the ceil function works"""
        a = gpuarray.arange(test_sample, dtype=numpy.float32)/10        
        b = cumath.ceil(a)
        
        a = a.get()
        b = b.get()
        
        for i in range(test_sample):
            self.assert_(math.ceil(a[i]) == b[i])
               
    def test_fabs(self):
        """tests if the fabs function works"""
        a = gpuarray.arange(test_sample, dtype=numpy.float32) * -1    
        b = cumath.fabs(a)
        
        a = a.get()
        b = b.get()
        
        for i in range(test_sample):
            self.assert_(a[i] + b[i] == 0)
            self.assert_(b[i] >= 0)
               
    def test_floor(self):
        """tests if the floor function works"""
        a = gpuarray.arange(test_sample, dtype=numpy.float32)/10        
        b = cumath.floor(a)

        a = a.get()
        b = b.get()
        
        for i in range(test_sample):
            self.assert_(math.floor(a[i]) == b[i])
               
    def test_fmod(self):
        """tests if the fmod function works"""
        a = gpuarray.arange(test_sample, dtype=numpy.float32)/10        
        a2 = gpuarray.arange(test_sample, dtype=numpy.float32)/45.2 + 0.1
        b = cumath.fmod(a, a2)
        
        a = a.get()
        a2 = a2.get()
        b = b.get()
        
        for i in range(test_sample):
            self.assert_(math.fmod(a[i], a2[i]) == b[i])

    def test_ldexp(self):
        """tests if the ldexp function works"""
        a = gpuarray.arange(test_sample, dtype=numpy.float32)       
        a2 = gpuarray.arange(test_sample, dtype=numpy.float32)*1e-3
        b = cumath.ldexp(a,a2)
        
        a = a.get()
        a2 = a2.get()
        b = b.get()
        
        for i in range(test_sample):
            self.assert_(math.ldexp(a[i], int(a2[i])) == b[i])
               
    def test_modf(self):
        """tests if the modf function works"""
        a = gpuarray.arange(test_sample, dtype=numpy.float32)/10  
        fracpart, intpart = cumath.modf(a)
    
        a = a.get()
        intpart = intpart.get()
        fracpart = fracpart.get()
        
        for i in range(test_sample):
            fracpart_true, intpart_true = math.modf(a[i])
            
            self.assert_(intpart_true == intpart[i])
            self.assert_(abs(fracpart_true - fracpart[i]) < 1e-4)
                         
    def test_frexp(self):
        """tests if the frexp function works"""
        a = gpuarray.arange(test_sample, dtype=numpy.float32)/10  
        significands, exponents = cumath.frexp(a)
    
        a = a.get()
        significands = significands.get()
        exponents = exponents.get()
        
        for i in range(test_sample):
            sig_true, ex_true = math.frexp(a[i])
            
            self.assert_(sig_true == significands[i])
            self.assert_(ex_true == exponents[i])

    def test_exp(self):
        """tests if the exp function works"""
        a = gpuarray.arange(100, dtype=numpy.float32)/10        
        b = cumath.exp(a)
        
        a = a.get()
        b = b.get()
        
        for i in range(100):
            self.assert_(abs(math.exp(a[i]) - b[i]) < 1e-2)

    def test_log(self):
        """tests if the log function works"""
        a = gpuarray.arange(test_sample, dtype=numpy.float32)+1       
        b = cumath.log(a)
        
        a = a.get()
        b = b.get()
        
        for i in range(test_sample):
            self.assert_(abs(math.log(a[i]) - b[i]) < 1e-3)

    def test_log10(self):
        """tests if the log function works"""
        a = gpuarray.arange(test_sample, dtype=numpy.float32)+1       
        b = cumath.log10(a)
        
        a = a.get()
        b = b.get()
        
        for i in range(test_sample):
            self.assert_(abs(math.log10(a[i]) - b[i]) < 1e-3)
         
    def test_pow(self):
        """tests if the pow function works"""
        a = gpuarray.arange(10, dtype=numpy.float32)+1       
        b = a**2
        
        a = a.get()
        b = b.get()
        
        for i in range(10):
            self.assert_(abs(math.pow(a[i],2) - b[i]) < 1e-3)
         
    def test_sqrt(self):
        """tests if the sqrt function works"""
        a = gpuarray.arange(10, dtype=numpy.float32)+1       
        b = cumath.sqrt(a)
        
        a = a.get()
        b = b.get()
        
        for i in range(10):
            self.assert_(abs(math.sqrt(a[i]) - b[i]) < 1e-3)
         
    def test_acos(self):
        """tests if the acos function works"""
        a = (gpuarray.arange(10, dtype=numpy.float32)+1)/100      
        b = cumath.acos(a)
        
        a = a.get()
        b = b.get()
        
        for i in range(10):
            self.assert_(abs(math.acos(a[i]) - b[i]) < 1e-2)
          
    def test_cos(self):
        """tests if the cos function works"""
        a = (gpuarray.arange(10, dtype=numpy.float32)+1)/100      
        b = cumath.cos(a)
        
        a = a.get()
        b = b.get()
        
        for i in range(10):
            self.assert_(abs(math.cos(a[i]) - b[i]) < 1e-2)         
         
    def test_sin(self):
        """tests if the sin function works"""
        a = (gpuarray.arange(10, dtype=numpy.float32)+1)/100      
        b = cumath.sin(a)
        
        a = a.get()
        b = b.get()
        
        for i in range(10):
            self.assert_(abs(math.sin(a[i]) - b[i]) < 1e-2)
          
    def test_asin(self):
        a = (gpuarray.arange(10, dtype=numpy.float32)+1)/100      
        b = cumath.asin(a)
        
        a = a.get()
        b = b.get()
        
        for i in range(10):
            self.assert_(abs(math.asin(a[i]) - b[i]) < 1e-2)         

         
    def test_tan(self):
        a = (gpuarray.arange(10, dtype=numpy.float32)+1)/100      
        b = cumath.tan(a)
        
        a = a.get()
        b = b.get()
        
        for i in range(10):
            self.assert_(abs(math.tan(a[i]) - b[i]) < 1e-2)
          
    def test_atan(self):
        a = (gpuarray.arange(10, dtype=numpy.float32)+1)/100      
        b = cumath.atan(a)
        
        a = a.get()
        b = b.get()
        
        for i in range(10):
            self.assert_(abs(math.atan(a[i]) - b[i]) < 1e-2)         

    def test_cosh(self):
        a = (gpuarray.arange(10, dtype=numpy.float32)+1)/100      
        b = cumath.cosh(a)
        
        a = a.get()
        b = b.get()
        
        for i in range(10):
            self.assert_(abs(math.cosh(a[i]) - b[i]) < 1e-2)         
         
    def test_sinh(self):
        a = (gpuarray.arange(10, dtype=numpy.float32)+1)/100      
        b = cumath.sinh(a)
        
        a = a.get()
        b = b.get()
        
        for i in range(10):
            self.assert_(abs(math.sinh(a[i]) - b[i]) < 1e-2)

         
    def test_tanh(self):
        a = (gpuarray.arange(10, dtype=numpy.float32)+1)/100      
        b = cumath.tanh(a)
        
        a = a.get()
        b = b.get()
        
        for i in range(10):
            self.assert_(abs(math.tanh(a[i]) - b[i]) < 1e-2)




if __name__ == '__main__':
    unittest.main()
