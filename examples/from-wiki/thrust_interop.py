#!python 

import pycuda
import pycuda.autoinit
import pycuda.gpuarray as gpuarray
import numpy as np

from cgen import *
from codepy.bpl import BoostPythonModule
from codepy.cuda import CudaModule

#Make a host_module, compiled for CPU
host_mod = BoostPythonModule()

#Make a device module, compiled with NVCC
nvcc_mod = CudaModule(host_mod)

#Describe device module code
#NVCC includes
nvcc_includes = [
    'thrust/sort.h',
    'thrust/device_vector.h',
    'cuda.h',
    ]
#Add includes to module
nvcc_mod.add_to_preamble([Include(x) for x in nvcc_includes])

#NVCC function
nvcc_function = FunctionBody(
    FunctionDeclaration(Value('void', 'my_sort'),
                        [Value('CUdeviceptr', 'input_ptr'),
                         Value('int', 'length')]),
    Block([Statement('thrust::device_ptr<float> thrust_ptr((float*)input_ptr)'),
           Statement('thrust::sort(thrust_ptr, thrust_ptr+length)')]))

#Add declaration to nvcc_mod
#Adds declaration to host_mod as well
nvcc_mod.add_function(nvcc_function)

host_includes = [
    'boost/python/extract.hpp',
    ]
#Add host includes to module
host_mod.add_to_preamble([Include(x) for x in host_includes])

host_namespaces = [
    'using namespace boost::python',
    ]

#Add BPL using statement
host_mod.add_to_preamble([Statement(x) for x in host_namespaces])


host_statements = [
    #Extract information from PyCUDA GPUArray
    #Get length
    'tuple shape = extract<tuple>(gpu_array.attr("shape"))',
    'int length = extract<int>(shape[0])',
    #Get data pointer
    'CUdeviceptr ptr = extract<CUdeviceptr>(gpu_array.attr("ptr"))',
    #Call Thrust routine, compiled into the CudaModule
    'my_sort(ptr, length)',
    #Return result
    'return gpu_array',
    ]

host_mod.add_function(
    FunctionBody(
        FunctionDeclaration(Value('object', 'host_entry'),
                            [Value('object', 'gpu_array')]),
        Block([Statement(x) for x in host_statements])))

#Print out generated code, to see what we're actually compiling
print("---------------------- Host code ----------------------")
print((host_mod.generate()))
print("--------------------- Device code ---------------------")
print((nvcc_mod.generate()))
print("-------------------------------------------------------")



#Compile modules
import codepy.jit, codepy.toolchain
gcc_toolchain = codepy.toolchain.guess_toolchain()
nvcc_toolchain = codepy.toolchain.guess_nvcc_toolchain()

module = nvcc_mod.compile(gcc_toolchain, nvcc_toolchain, debug=True)



length = 100
a = np.array(np.random.rand(length), dtype=np.float32)
print("---------------------- Unsorted -----------------------")
print(a)
b = gpuarray.to_gpu(a)
# Call Thrust!!
c = module.host_entry(b)
print("----------------------- Sorted ------------------------")
print(c.get())
print("-------------------------------------------------------")

