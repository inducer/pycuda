#include <boost/python.hpp>
#include <pycuda/pycuda.hpp>




using namespace pycuda;
namespace py = boost::python;
namespace ublas = boost::numeric::ublas;




namespace
{
  template <class T>
  void expose_device_ptr(std::string const &tp_name)
  {
    typedef pycuda::typed_device_ptr<T> cl;
    py::class_<cl>(
        ("DevicePtr"+tp_name).c_str(), py::no_init)
      ;
  }
}




BOOST_PYTHON_MODULE(_rt)
{
  expose_device_ptr<float>("Float32");
}

