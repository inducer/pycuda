import pycuda.simplearray as simplearray
import unittest
import pycuda.math as cumath
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
          
               
if __name__ == '__main__':
    unittest.main()