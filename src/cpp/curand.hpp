#ifndef _AFJDFJSDFSD_PYCUDA_HEADER_SEEN_CURAND_HPP
#define _AFJDFJSDFSD_PYCUDA_HEADER_SEEN_CURAND_HPP


#if CUDAPP_CUDA_VERSION >= 3020
#include <curand.h>
#endif


namespace pycuda { namespace curandom {

  py::tuple py_curand_version()
  {
    int version = 0;
#if CUDAPP_CUDA_VERSION >= 3020
    curandGetVersion(&version);
#endif
    return py::make_tuple(
        version / 1000,
        (version % 1000)/10,
        version % 10);
  }

#if CUDAPP_CUDA_VERSION >= 3020
  void py_curand_get_direction_vectors32(curandDirectionVectors32_t *vectors[],
      curandDirectionVectorSet_t set)
// TODO: checking; cannot use CUDAPP_CALL_GUARDED because function returns CURAND enum
  { curandGetDirectionVectors32(vectors, set); }
#endif

} }

#endif

