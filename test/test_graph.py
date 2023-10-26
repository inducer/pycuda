__copyright__ = """
Copyright 2008-2021 Andreas Kloeckner
Copyright 2021 NVIDIA Corporation
"""

import numpy as np
import numpy.linalg as la
from pycuda.tools import mark_cuda_test, dtype_to_ctype
import pytest  # noqa


import pycuda.gpuarray as gpuarray
import pycuda.driver as drv
from pycuda.compiler import SourceModule


class TestGraph:
    @mark_cuda_test
    def test_static_params(self):
        mod = SourceModule("""
            __global__ void plus(float *a, int num)
            {
            int idx = threadIdx.x + threadIdx.y*4;
            a[idx] += num;
            }
            """)
        func_plus = mod.get_function("plus")

        import numpy
        a = numpy.zeros((4, 4)).astype(numpy.float32)
        a_gpu = drv.mem_alloc_like(a)
        result = numpy.zeros_like(a)
        stream_1 = drv.Stream()
        stream_1.begin_capture()
        func_plus(a_gpu, numpy.int32(3), block=(4, 4, 1), stream=stream_1)
        graph = stream_1.end_capture()

        instance = graph.instantiate()
        instance.launch()
        drv.memcpy_dtoh_async(result, a_gpu, stream_1)
        np.testing.assert_allclose(result, np.full((4, 4), 3), rtol=1e-5)

    @mark_cuda_test
    def test_dynamic_params(self):
        mod = SourceModule("""
            __global__ void plus(float *a, int num)
            {
            int idx = threadIdx.x + threadIdx.y*4;
            a[idx] += num;
            }
            """)
        func_plus = mod.get_function("plus")

        stream_1 = drv.Stream()
        import numpy
        a = numpy.zeros((4, 4)).astype(numpy.float32)
        a_gpu = drv.mem_alloc_like(a)
        result = numpy.zeros_like(a)
        stream_1.begin_capture()
        stat , _, x_graph, deps = stream_1.get_capture_info_v2()
        assert stat == drv.capture_status.ACTIVE, "Capture should be active"
        assert len(deps) == 0, "Nothing on deps"
        newnode = x_graph.add_kernel_node(a_gpu, numpy.int32(3), block=(4, 4, 1), func=func_plus, dependencies=deps)
        stream_1.update_capture_dependencies([newnode], drv.update_capture_dependencies_flags.SET_CAPTURE_DEPENDENCIES)
        drv.memcpy_dtoh_async(result, a_gpu, stream_1) # Capture a copy as well.
        graph = stream_1.end_capture()
        assert graph == x_graph, "Should be the same"

        instance = graph.instantiate()

        stat, _, _, _ = stream_1.get_capture_info_v2()
        assert stat == drv.capture_status.NONE, "No capture should be active"

        wanna = 0
        for i in range(4):
            instance.kernel_node_set_params(a_gpu, numpy.int32(i), block=(4, 4, 1), func=func_plus, kernel_node=newnode)
            instance.launch()
            wanna += i
            np.testing.assert_allclose(result, np.full((4, 4), wanna), rtol=1e-5)

    @mark_cuda_test
    def test_many_dynamic_params(self):
        mod = SourceModule("""
            __global__ void plus(float *a, int num)
            {
            int idx = threadIdx.x + threadIdx.y*4;
            a[idx] += num;
            }
            __global__ void times(float *a, float *b)
            {
            int idx = threadIdx.x + threadIdx.y*4;
            a[idx] *= b[idx];
            }
            """)
        func_plus = mod.get_function("plus")
        func_times = mod.get_function("times")

        stream_1 = drv.Stream()
        _ , _, _, _ = stream_1.get_capture_info_v2()

        import numpy
        a = numpy.zeros((4, 4)).astype(numpy.float32)
        a_gpu = drv.mem_alloc_like(a)
        b = numpy.zeros((4, 4)).astype(numpy.float32)
        b_gpu = drv.mem_alloc_like(b)
        result = numpy.zeros_like(b)
        _ , _, _, _ = stream_1.get_capture_info_v2()
        stream_1.begin_capture()
        stat , _, x_graph, deps = stream_1.get_capture_info_v2()
        assert stat == drv.capture_status.ACTIVE, "Capture should be active"
        assert len(deps) == 0, "Nothing on deps"
        newnode = x_graph.add_kernel_node(a_gpu, numpy.int32(3), block=(4, 4, 1), func=func_plus, dependencies=deps)
        stream_1.update_capture_dependencies([newnode], drv.update_capture_dependencies_flags.SET_CAPTURE_DEPENDENCIES)
        _, _, x_graph, deps = stream_1.get_capture_info_v2()
        assert deps == [newnode], "Call to update_capture_dependencies should set newnode as the only dep"
        newnode2 = x_graph.add_kernel_node(b_gpu, numpy.int32(3), block=(4, 4, 1), func=func_plus, dependencies=deps)
        stream_1.update_capture_dependencies([newnode2], drv.update_capture_dependencies_flags.SET_CAPTURE_DEPENDENCIES)

        # Static capture
        func_times(a_gpu, b_gpu, block=(4, 4, 1), stream=stream_1)
        drv.memcpy_dtoh_async(result, a_gpu, stream_1) # Capture a copy as well.
        graph = stream_1.end_capture()
        assert graph == x_graph, "Should be the same"

        instance = graph.instantiate()

        stat, _, _, _ = stream_1.get_capture_info_v2()
        assert stat == drv.capture_status.NONE, "No capture be active"

        instance.kernel_node_set_params(a_gpu, numpy.int32(4), block=(4, 4, 1), func=func_plus, kernel_node=newnode)
        instance.kernel_node_set_params(b_gpu, numpy.int32(9), block=(4, 4, 1), func=func_plus, kernel_node=newnode2)
        instance.launch()
        np.testing.assert_allclose(result, np.full((4, 4), 4*9), rtol=1e-5)

        a = numpy.zeros((4, 4)).astype(numpy.float32)
        a_gpu_fake = drv.mem_alloc_like(a)
        instance.kernel_node_set_params(a_gpu_fake, numpy.int32(5), block=(4, 4, 1), func=func_plus, kernel_node=newnode)
        instance.kernel_node_set_params(b_gpu, numpy.int32(4), block=(4, 4, 1), func=func_plus, kernel_node=newnode2)
        instance.launch()
        np.testing.assert_allclose(result, np.full((4, 4), (4*9)*(9+4)), rtol=1e-5) # b is now (9 + 4), a is 4*9 as it was after func_times, since we write to another buffer this launch.

    @mark_cuda_test
    def test_graph_create(self):
        mod = SourceModule("""
            __global__ void plus(float *a, int num)
            {
            int idx = threadIdx.x + threadIdx.y*4;
            a[idx] += num;
            }
            """)
        func_plus = mod.get_function("plus")

        import numpy
        a = numpy.zeros((4, 4)).astype(numpy.float32)
        a_gpu = drv.mem_alloc_like(a)
        result = numpy.zeros_like(a)

        graph = drv.Graph()
        node1 = graph.add_kernel_node(a_gpu, numpy.int32(1), block=(4, 4, 1), func=func_plus, dependencies=[])
        node2 = graph.add_kernel_node(a_gpu, numpy.int32(2), block=(4, 4, 1), func=func_plus, dependencies=[node1])

        instance = graph.instantiate()
        instance.launch()
        drv.memcpy_dtoh_async(result, a_gpu)
        np.testing.assert_allclose(result, np.full((4, 4), 1+2), rtol=1e-5)

if __name__ == "__main__":
    # make sure that import failures get reported, instead of skipping the tests.
    import pycuda.autoinit  # noqa

    import sys

    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        from pytest import main

        main([__file__])
