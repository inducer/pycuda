#!python 
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule

mod = SourceModule("""
    #include <stdio.h>

    __global__ void say_hi()
    {
      printf("I am %d.%d\\n", threadIdx.x, threadIdx.y);
    }
    """)

func = mod.get_function("say_hi")
func(block=(4,4,1))

