#include <boost/python.hpp>




void pycuda_wrap_cublas();
void pycuda_wrap_cudart();




BOOST_PYTHON_MODULE(_internal)
{
  pycuda_wrap_cublas();
  pycuda_wrap_cudart();
}

