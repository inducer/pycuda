import pycuda.driver as drv
import pycuda.tools
import pycuda.autoinit
import numpy
import numpy.linalg as la
from pycuda.compiler import SourceModule

mod = SourceModule("""
   attributes (global) subroutine multiply_them(dest,a,b) bind(c)
   use, intrinsic :: iso_fortran_env
   implicit none
   real(kind=real32), intent(out) :: dest(*)
   real(kind=real32), intent(in) :: a(*),b(*)
   integer(kind=int32) :: i
   i=threadIdx%x
   dest(i)=a(i)*b(i)
   end subroutine multiply_them
""",nvcc='nvfortran',no_extern_c=True)
#""",nvcc='nvfortran',no_extern_c=True,options=["-O3", "-fast"])

multiply_them = mod.get_function("multiply_them")

a = numpy.random.randn(400).astype(numpy.float32)
b = numpy.random.randn(400).astype(numpy.float32)

dest = numpy.zeros_like(a)
multiply_them(
        drv.Out(dest), drv.In(a), drv.In(b),
        block=(400,1,1))

print(dest-a*b)
