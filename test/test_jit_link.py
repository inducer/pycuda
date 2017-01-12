import sys
from pycuda.tools import mark_cuda_test

def have_pycuda():
    try:
        import pycuda  # noqa
        return True
    except:
        return False

if have_pycuda():
    import pycuda.driver as drv  # noqa
    from pycuda.compiler import SourceModule
    from pycuda.compiler import JitLinkModule
    from pycuda.driver import jit_input_type

class TestJitLink:
    @mark_cuda_test
    def test_create(self):
        mod = JitLinkModule()

    @mark_cuda_test
    def test_static_parallelism(self):
        test_cu = '''#include <cstdio>
        __global__ void test_kernel() {
            printf("Hello world!\\n");
        }'''

        mod = SourceModule(test_cu)
        test_kernel = mod.get_function('test_kernel')
        test_kernel(grid=(2,1), block=(1,1,1))

    @mark_cuda_test
    def test_dynamic_parallelism(self):
        # nvcc error:
        #     calling a __global__ function("test_kernel_inner") from a
        #         __global__ function("test_kernel") is only allowed on the
        #     compute_35 architecture or above
        import pycuda.autoinit
        compute_capability = pycuda.autoinit.device.compute_capability()
        if compute_capability[0] < 3 or (compute_capability[0] == 3 and compute_capability[1] < 5):
            raise Exception('Minimum compute capability for dynamic parallelism is 3.5 (found: %u.%u)!' %
                (compute_capability[0], compute_capability[1]))

        import os, os.path
        from platform import system
        if system() == 'Windows':
            cudadevrt = os.path.join(os.environ['CUDA_PATH'], 'lib/x64/cudadevrt.lib')
        else:
            cudadevrt = '/usr/lib/x86_64-linux-gnu/libcudadevrt.a'  # TODO: this is just an untested guess!
        if not os.path.isfile(cudadevrt):
            raise Exception('Cannot locate library cudadevrt!')

        test_cu = '''#include <cstdio>
        __global__ void test_kernel_inner() {
            printf("  Hello inner world!\\n");
        }
        __global__ void test_kernel() {
            printf("Hello outer world!\\n");
            test_kernel_inner<<<2, 1>>>();
        }'''

        mod = JitLinkModule()
        mod.add_source(test_cu, nvcc_options=['-rdc=true', '-lcudadevrt'])
        mod.add_file(cudadevrt, jit_input_type.LIBRARY)
        mod.link()
        test_kernel = mod.get_function('test_kernel')
        test_kernel(grid=(2,1), block=(1,1,1))

if __name__ == "__main__":
    # make sure that import failures get reported, instead of skipping the tests.
    import pycuda.autoinit  # noqa

    if len(sys.argv) > 1:
        exec (sys.argv[1])
    else:
        from py.test.cmdline import main
        main([__file__])
