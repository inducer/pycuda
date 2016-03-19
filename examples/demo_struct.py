# prepared invocations and structures -----------------------------------------
from __future__ import print_function
from __future__ import absolute_import
import pycuda.driver as cuda
import pycuda.autoinit
import numpy
from pycuda.compiler import SourceModule

class DoubleOpStruct:
    mem_size = 8 + numpy.uintp(0).nbytes
    def __init__(self, array, struct_arr_ptr):
        self.data = cuda.to_device(array)
        self.shape, self.dtype = array.shape, array.dtype
        """
        numpy.getbuffer() needed due to lack of new-style buffer interface for
        scalar numpy arrays as of numpy version 1.9.1

        see: https://github.com/inducer/pycuda/pull/60
        """
        cuda.memcpy_htod(int(struct_arr_ptr),
                         numpy.getbuffer(numpy.int32(array.size)))
        cuda.memcpy_htod(int(struct_arr_ptr) + 8,
                         numpy.getbuffer(numpy.uintp(int(self.data))))

    def __str__(self):
        return str(cuda.from_device(self.data, self.shape, self.dtype))

struct_arr = cuda.mem_alloc(2 * DoubleOpStruct.mem_size)
do2_ptr = int(struct_arr) + DoubleOpStruct.mem_size

array1 = DoubleOpStruct(numpy.array([1, 2, 3], dtype=numpy.float32), struct_arr)
array2 = DoubleOpStruct(numpy.array([0, 4], dtype=numpy.float32), do2_ptr)

print("original arrays")
print(array1)
print(array2)

mod = SourceModule("""
    struct DoubleOperation {
        int datalen, __padding; // so 64-bit ptrs can be aligned
        float *ptr;
    };


    __global__ void double_array(DoubleOperation *a)
    {
        a = a + blockIdx.x;
        for (int idx = threadIdx.x; idx < a->datalen; idx += blockDim.x)
        {
            float *a_ptr = a->ptr;
            a_ptr[idx] *= 2;
        }
    }
    """)
func = mod.get_function("double_array")
func(struct_arr, block=(32, 1, 1), grid=(2, 1))

print("doubled arrays")
print(array1)
print(array2)

func(numpy.uintp(do2_ptr), block=(32, 1, 1), grid=(1, 1))
print("doubled second only")
print(array1)
print(array2)

if cuda.get_version() < (4, ):
    func.prepare("P", block=(32, 1, 1))
    func.prepared_call((2, 1), struct_arr)
else:
    func.prepare("P")
    block = (32, 1, 1)
    func.prepared_call((2, 1), block, struct_arr)


print("doubled again")
print(array1)
print(array2)

if cuda.get_version() < (4, ):
    func.prepared_call((1, 1), do2_ptr)
else:
    func.prepared_call((1, 1), block, do2_ptr)


print("doubled second only again")
print(array1)
print(array2)
