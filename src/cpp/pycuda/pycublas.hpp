#ifndef HEADER_SEEN_AKFYHRSDFAA_PYCUDA_HPP
#define HEADER_SEEN_AKFYHRSDFAA_PYCUDA_HPP




#include <cublas.h>
#include <pyublas/numpy.hpp>
#include <boost/numeric/bindings/traits/traits.hpp>




#define PYCUDA_CUBLAS_CALL_HELPER(NAME, ARGLIST) \
  { \
    cublasStatus cu_status_code = NAME ARGLIST; \
    if (cu_status_code != CUBLAS_STATUS_SUCCESS) \
      throw std::runtime_error(#NAME " failed: "+get_cublas_error_str(cu_status_code));\
  }
#define PYCUDA_CUBLAS_CHECK_ERROR \
  PYCUDA_CUBLAS_CALL_HELPER(cublasGetError, ());




namespace pycuda
{
  std::string get_cublas_error_str(cublasStatus s)
  {
    switch (s)
    {
      case CUBLAS_STATUS_SUCCESS: 
        return "operation completed successfully";
      case CUBLAS_STATUS_NOT_INITIALIZED:
        return "CUBLAS library not initialized";
      case CUBLAS_STATUS_ALLOC_FAILED:
        return "resource allocation failed";
      case CUBLAS_STATUS_INVALID_VALUE:
        return "unsupported numerical value was passed to function";
      case CUBLAS_STATUS_ARCH_MISMATCH:
        return "function requires an architectural feature absent "
          "from the architecture of the device";
      case CUBLAS_STATUS_MAPPING_ERROR:
        return "access to GPU memory space failed";
      case CUBLAS_STATUS_EXECUTION_FAILED:
        return "GPU program failed to execute";
      case CUBLAS_STATUS_INTERNAL_ERROR:
        return "an internal CUBLAS operation failed";
      default:
        return "cublas: unkown error";
    }
  }




  template<class T>
  struct typed_device_ptr
  {
    T *data;
  };




  template<class T>
  struct cublas_device_ptr : public typed_device_ptr<T>
  {
    cublas_device_ptr(int n)
    {
      PYCUDA_CUBLAS_CALL_HELPER(cublasAlloc, (n, sizeof(T), 
            reinterpret_cast<void **>(&this->data)));
    }

    ~cublas_device_ptr()
    {
      PYCUDA_CUBLAS_CALL_HELPER(cublasFree, (this->data));
    }

    template<class VecType>
    void set(const VecType &v, int cpu_spacing=1, int gpu_spacing=1)
    {
      using namespace boost::numeric::bindings;
      PYCUDA_CUBLAS_CALL_HELPER(cublasSetVector, (
          v.size()/cpu_spacing,
          sizeof(T),
          traits::vector_storage(v), 
          cpu_spacing,
          this->data,
          gpu_spacing
          ));

    }

    void set(const pyublas::numpy_vector<T> v, int gpu_spacing=1)
    {
      set(v, v.min_stride()/sizeof(T), gpu_spacing);
    }

    template<class VecType>
    void get(VecType &v, int cpu_spacing=1, int gpu_spacing=1)
    {
      using namespace boost::numeric::bindings;
      PYCUDA_CUBLAS_CALL_HELPER(cublasGetVector, (
          v.size()/cpu_spacing,
          sizeof(T),
          this->data,
          gpu_spacing,
          traits::vector_storage(v), 
          cpu_spacing
          ));
    }

    void get(pyublas::numpy_vector<T> v, int gpu_spacing=1)
    {
      get(v, v.min_stride()/sizeof(T), gpu_spacing);
    }
  };




  namespace blas
  {
    inline
    void axpy(int n, float alpha, 
        cublas_device_ptr<float> const &x, int incx, 
        cublas_device_ptr<float> &y, int incy)
    {
      cublasSaxpy(n, alpha, x.data, incx, y.data, incy);
      PYCUDA_CUBLAS_CHECK_ERROR;
    }

    inline
    void copy(int n, 
        cublas_device_ptr<float> const &x, int incx, 
        cublas_device_ptr<float> &y, int incy)
    {
      cublasScopy(n, x.data, incx, y.data, incy);
      PYCUDA_CUBLAS_CHECK_ERROR;
    }

    inline
    float dot(int n, 
        cublas_device_ptr<float> const &x, int incx, 
        cublas_device_ptr<float> &y, int incy)
    {
      float result = cublasSdot(n, x.data, incx, y.data, incy);
      PYCUDA_CUBLAS_CHECK_ERROR;
      return result;
    }

    inline
    float nrm2(int n, cublas_device_ptr<float> const &x, int incx)
    {
      float result = cublasSnrm2(n, x.data, incx);
      PYCUDA_CUBLAS_CHECK_ERROR;
      return result;
    }

    inline 
    void scal(int n, float alpha, cublas_device_ptr<float> const &x, int incx)
    {
      cublasSscal(n, alpha, x.data, incx);
      PYCUDA_CUBLAS_CHECK_ERROR;
    }

    inline
    void swap(int n, 
        cublas_device_ptr<float> const &x, int incx, 
        cublas_device_ptr<float> &y, int incy)
    {
      cublasSswap(n, x.data, incx, y.data, incy);
      PYCUDA_CUBLAS_CHECK_ERROR;
    }
  }
}




#endif
