#include <cuda.hpp>
#include <curand.hpp>

#include "tools.hpp"
#include "wrap_helpers.hpp"

#if CUDAPP_CUDA_VERSION >= 3020
#include <curand.h>
#endif

using namespace pycuda;
using namespace pycuda::curandom;

void pycuda_expose_curand()
{
  using py::arg;
  using py::args;

#if CUDAPP_CUDA_VERSION >= 3020
  py::enum_<curandDirectionVectorSet_t>("direction_vector_set")
    .value("VECTOR_32", CURAND_DIRECTION_VECTORS_32_JOEKUO6)
#if CUDAPP_CUDA_VERSION >= 4000
    .value("SCRAMBLED_VECTOR_32", CURAND_SCRAMBLED_DIRECTION_VECTORS_32_JOEKUO6)
    .value("VECTOR_64", CURAND_DIRECTION_VECTORS_64_JOEKUO6)
    .value("SCRAMBLED_VECTOR_64", CURAND_SCRAMBLED_DIRECTION_VECTORS_64_JOEKUO6)
#endif
  ;
#endif

  py::def("get_curand_version", py_curand_version);

#if CUDAPP_CUDA_VERSION >= 3020
  py::def("_get_direction_vectors", py_curand_get_direction_vectors,
      (arg("set"), arg("dst"), arg("count")));
#endif

#if CUDAPP_CUDA_VERSION >= 4000
  py::def("_get_scramble_constants32", py_curand_get_scramble_constants32,
      (arg("dst"), arg("count")));
  py::def("_get_scramble_constants64", py_curand_get_scramble_constants64,
      (arg("dst"), arg("count")));
#endif
}
