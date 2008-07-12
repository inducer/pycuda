import pycuda.simplearray as simplearray
import unittest
import pycuda.cumath as cumath
import math as math

class TestMath(unittest.TestCase):
    
    def test_ceil(self):
        """tests if the ceil function works"""
        a = simplearray.array(100).fill_arange()/10        
        b = cumath.ceil(a)
        
        for i in range(100):
            self.assert_(math.ceil(a[i]) == b[i])
               
    def test_fabs(self):
        """tests if the fabs function works"""
        a = simplearray.array(100).fill_arange() * -1    
        b = cumath.fabs(a)
        
        for i in range(100):
            self.assert_(a[i] + b[i] == 0)
            self.assert_(b[i] >= 0)
               
    def test_floor(self):
        """tests if the floor function works"""
        a = simplearray.array(100).fill_arange()/10        
        b = cumath.floor(a)
        
        for i in range(100):
            self.assert_(math.floor(a[i]) == b[i])
               
    def test_fmod(self):
        """tests if the fmod function works"""
        a = simplearray.array(100).fill_arange()/10        
        b = cumath.fmod(a,2)
        
        for i in range(100):
            self.assert_(math.fmod(a[i],2) == b[i])

    def test_ldexp(self):
        """tests if the ldexp function works"""
        a = simplearray.array(100).fill_arange()       
        b = cumath.ldexp(a,5)
        
        for i in range(100):
            self.assert_(math.ldexp(a[i],5) == b[i])
               
    def test_modf(self):
        """tests if the modf function works"""
        a = simplearray.array(100).fill_arange()/10  
        b = cumath.modf(a)
    
        first = b[0]
        second = b[1]    
        
        for i in range(100):
            c = math.modf(a[i])
            
            self.assert_(c[0] == first[i])
            self.assert_(c[1] == second[i])
                         
    def test_frexp(self):
        """tests if the frexp function works"""
        a = simplearray.array(100).fill_arange()/10  
        b = cumath.frexp(a)
    
        first = b[0]
        second = b[1]    
        
        for i in range(100):
            c = math.frexp(a[i])
            
            self.assert_(c[0] == first[i])
            self.assert_(c[1] == second[i])

    def test_exp(self):
        """tests if the exp function works"""
        a = simplearray.array(100).fill_arange()/10        
        b = cumath.exp(a)
        
        for i in range(100):
            self.assert_(abs(math.exp(a[i]) - b[i]) < 1e-2)

    def test_log(self):
        """tests if the log function works"""
        a = simplearray.array(100).fill_arange()+1       
        b = cumath.log(a)
        
        for i in range(100):
            self.assert_(abs(math.log(a[i]) - b[i]) < 1e-3)

    def test_log10(self):
        """tests if the log function works"""
        a = simplearray.array(100).fill_arange()+1       
        b = cumath.log10(a)
        
        for i in range(100):
            self.assert_(abs(math.log10(a[i]) - b[i]) < 1e-3)
         
    def test_pow(self):
        """tests if the pow function works"""
        a = simplearray.array(10).fill_arange()+1       
        b = cumath.pow(a,2)
        
        for i in range(10):
            self.assert_(abs(math.pow(a[i],2) - b[i]) < 1e-3)
         
    def test_sqrt(self):
        """tests if the sqrt function works"""
        a = simplearray.array(10).fill_arange()+1       
        b = cumath.sqrt(a)
        
        for i in range(10):
            self.assert_(abs(math.sqrt(a[i]) - b[i]) < 1e-3)
         
    def test_acos(self):
        """tests if the acos function works"""
        a = (simplearray.array(10).fill_arange()+1)/100      
        b = cumath.acos(a)
        
        for i in range(10):
            self.assert_(abs(math.acos(a[i]) - b[i]) < 1e-2)
          
    def test_cos(self):
        """tests if the cos function works"""
        a = (simplearray.array(10).fill_arange()+1)/100      
        b = cumath.cos(a)
        
        for i in range(10):
            self.assert_(abs(math.cos(a[i]) - b[i]) < 1e-2)         
         
    def test_sin(self):
        """tests if the sin function works"""
        a = (simplearray.array(10).fill_arange()+1)/100      
        b = cumath.sin(a)
        
        for i in range(10):
            self.assert_(abs(math.sin(a[i]) - b[i]) < 1e-2)
          
    def test_asin(self):
        """tests if the asin function works"""
        a = (simplearray.array(10).fill_arange()+1)/100      
        b = cumath.asin(a)
        
        for i in range(10):
            self.assert_(abs(math.asin(a[i]) - b[i]) < 1e-2)         

         
    def test_tan(self):
        """tests if the tan function works"""
        a = (simplearray.array(10).fill_arange()+1)/100      
        b = cumath.tan(a)
        
        for i in range(10):
            self.assert_(abs(math.tan(a[i]) - b[i]) < 1e-2)
          
    def test_atan(self):
        """tests if the asin function works"""
        a = (simplearray.array(10).fill_arange()+1)/100      
        b = cumath.atan(a)
        
        for i in range(10):
            self.assert_(abs(math.atan(a[i]) - b[i]) < 1e-2)         

    def test_cosh(self):
        """tests if the cosh function works"""
        a = (simplearray.array(10).fill_arange()+1)/100      
        b = cumath.cosh(a)
        
        for i in range(10):
            self.assert_(abs(math.cosh(a[i]) - b[i]) < 1e-2)         
         
    def test_sinh(self):
        """tests if the sinh function works"""
        a = (simplearray.array(10).fill_arange()+1)/100      
        b = cumath.sinh(a)
        
        for i in range(10):
            self.assert_(abs(math.sinh(a[i]) - b[i]) < 1e-2)

         
    def test_tanh(self):
        """tests if the tanh function works"""
        a = (simplearray.array(10).fill_arange()+1)/100      
        b = cumath.tanh(a)
        
        for i in range(10):
            self.assert_(abs(math.tanh(a[i]) - b[i]) < 1e-2)

         
    def test_degrees(self):
        """tests if the degrees function works"""
        a = (simplearray.array(10).fill_arange()+1)     
        b = cumath.degrees(a)
        
        for i in range(10):
            self.assert_(abs(math.degrees(a[i]) - b[i]) < 1e-2)

         
    def test_radians(self):
        """tests if the radians function works"""
        a = (simplearray.array(10).fill_arange()+1)     
        b = cumath.radians(a)
        
        for i in range(10):
            self.assert_(abs(math.radians(a[i]) - b[i]) < 1e-2)
          
                    
if __name__ == '__main__':
    unittest.main()