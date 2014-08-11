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
    int n = 0;

    py_buffer_wrapper buf_wrapper;
    buf_wrapper.get(dst.ptr(), PyBUF_ANY_CONTIGUOUS | PyBUF_WRITABLE);

    void *buf = buf_wrapper.m_buf.buf;
    PYCUDA_BUFFER_SIZE_T len = buf_wrapper.m_buf.len;

    if (CURAND_DIRECTION_VECTORS_32_JOEKUO6 == set
#if CUDAPP_CUDA_VERSION >= 4000
      || CURAND_SCRAMBLED_DIRECTION_VECTORS_32_JOEKUO6 == set
#endif
    ) {
      curandDirectionVectors32_t *vectors;
      CURAND_CALL_GUARDED(curandGetDirectionVectors32, (&vectors, set));
      while (count > 0) {
        int size = ((count > 20000) ? 20000 : count)*sizeof(curandDirectionVectors32_t);
        memcpy((unsigned int *)buf+n*20000*sizeof(curandDirectionVectors32_t)/sizeof(unsigned int), vectors, size);
	count -= size/sizeof(curandDirectionVectors32_t);
        n++;
      }
    }
#if CUDAPP_CUDA_VERSION >= 4000
    if (CURAND_DIRECTION_VECTORS_64_JOEKUO6 == set
      || CURAND_SCRAMBLED_DIRECTION_VECTORS_64_JOEKUO6 == set) {
      curandDirectionVectors64_t *vectors;
      CURAND_CALL_GUARDED(curandGetDirectionVectors64, (&vectors, set));
      while (count > 0) {
        int size = ((count > 20000) ? 20000 : count)*sizeof(curandDirectionVectors64_t);
        memcpy((unsigned long long *)buf+n*20000*sizeof(curandDirectionVectors64_t)/sizeof(unsigned long long), vectors, size);
	count -= size/sizeof(curandDirectionVectors64_t);
        n++;
      }
    }
#endif
  }
#endif

#if CUDAPP_CUDA_VERSION >= 4000
  void py_curand_get_scramble_constants32(py::object dst, int count)
  {
    int n = 0;

    py_buffer_wrapper buf_wrapper;
    buf_wrapper.get(dst.ptr(), PyBUF_ANY_CONTIGUOUS | PyBUF_WRITABLE);

    void *buf = buf_wrapper.m_buf.buf;
    PYCUDA_BUFFER_SIZE_T len = buf_wrapper.m_buf.len;

    unsigned int *vectors;
    CURAND_CALL_GUARDED(curandGetScrambleConstants32, (&vectors));
// Documentation does not mention number of dimensions
// Assuming the same as in getDirectionVectors*
    while (count > 0) {
      int size = ((count > 20000) ? 20000 : count)*sizeof(unsigned int);
      memcpy((unsigned int *)buf+n*20000, vectors, size);
      count -= size/sizeof(unsigned int);
      n++;
    }
  }

  void py_curand_get_scramble_constants64(py::object dst, int count)
  {
    int n = 0;

    py_buffer_wrapper buf_wrapper;
    buf_wrapper.get(dst.ptr(), PyBUF_ANY_CONTIGUOUS | PyBUF_WRITABLE);

    void *buf = buf_wrapper.m_buf.buf;
    PYCUDA_BUFFER_SIZE_T len = buf_wrapper.m_buf.len;

    unsigned long long *vectors;
    CURAND_CALL_GUARDED(curandGetScrambleConstants64, (&vectors));
// Documentation does not mention number of dimensions
// Assuming the same as in getDirectionVectors*
    while (count > 0) {
      int size = ((count > 20000) ? 20000 : count)*sizeof(unsigned long long);
      memcpy((unsigned long long *)buf+n*20000, vectors, size);
      count -= size/sizeof(unsigned long long);
      n++;
    }
  }
#endif
} }

#endif

