#ifndef _AFJDFJSDFSD_PYCUDA_HEADER_SEEN_CURAND_HPP
#define _AFJDFJSDFSD_PYCUDA_HEADER_SEEN_CURAND_HPP


#if CUDAPP_CUDA_VERSION >= 3020
  #include <curand.h>

  #ifdef CUDAPP_TRACE_CUDA
    #define CURAND_PRINT_ERROR_TRACE(NAME, CODE) \
      if (CODE != CURAND_STATUS_SUCCESS) \
        std::cerr << NAME << " failed with code " << CODE << std::endl;
  #else
    #define CURAND_PRINT_ERROR_TRACE(NAME, CODE) /*nothing*/
  #endif

  #define CURAND_CALL_GUARDED(NAME, ARGLIST) \
    { \
      CUDAPP_PRINT_CALL_TRACE(#NAME); \
      curandStatus_t cu_status_code; \
      cu_status_code = NAME ARGLIST; \
      CURAND_PRINT_ERROR_TRACE(#NAME, cu_status_code); \
      if (cu_status_code != CURAND_STATUS_SUCCESS) \
        throw pycuda::error(#NAME, CUDA_SUCCESS);\
    }
#else
  #define CURAND_PRINT_ERROR_TRACE(NAME, CODE) /*nothing*/
  #define CURAND_CALL_GUARDED(NAME, ARGLIST) /*nothing*/
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
  void py_curand_get_direction_vectors(
      curandDirectionVectorSet_t set, py::object dst, int count)
  {
    void *buf;
    PYCUDA_BUFFER_SIZE_T len;
    int n = 0;

    if (PyObject_AsWriteBuffer(dst.ptr(), &buf, &len))
      throw py::error_already_set();
    if (CURAND_DIRECTION_VECTORS_32_JOEKUO6 == set) {
      curandDirectionVectors32_t *vectors;
      CURAND_CALL_GUARDED(curandGetDirectionVectors32, (&vectors, set));
      while (count > 0) {
        int size = ((count > 20000) ? 20000 : count)*sizeof(curandDirectionVectors32_t);
        memcpy((int *)buf+n*20000*sizeof(curandDirectionVectors32_t)/sizeof(unsigned int), vectors, size);
	count -= size/sizeof(curandDirectionVectors32_t);
        n++;
      }
    }
  }
#endif

} }

#endif

