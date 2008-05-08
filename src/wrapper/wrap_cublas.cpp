#include <boost/python.hpp>
#include <pycuda/pycuda.hpp>




using namespace pycuda;
namespace py = boost::python;
namespace ublas = boost::numeric::ublas;




namespace
{
  void init()
  { PYCUDA_CUBLAS_CALL_HELPER(cublasInit, ()); }
  void shutdown()
  { PYCUDA_CUBLAS_CALL_HELPER(cublasShutdown, ()); }




  template <class T>
  void expose_device_ptr(std::string const &tp_name)
  {
    typedef pycuda::cublas_device_ptr<T> cl;
    typedef pyublas::numpy_vector<T> vec;
    py::class_<cl, py::bases<pycuda::typed_device_ptr<T> > >(
        ("DevicePtr"+tp_name).c_str(),
        py::init<int>())
      .def("set", (void (cl::*)(vec const, int)) &cl::set,
          (py::arg("self"), py::arg("array"), py::arg("gpu_spacing")=1))
      .def("get", (void (cl::*)(vec, int)) &cl::get,
          (py::arg("self"), py::arg("array"), py::arg("gpu_spacing")=1))
      ;
  }
}




BOOST_PYTHON_MODULE(_blas)
{
  py::def("init", init);
  py::def("shutdown", shutdown);
  expose_device_ptr<float>("Float32");

  py::def("axpy", pycuda::blas::axpy);
  py::def("copy", pycuda::blas::copy);
  py::def("dot", pycuda::blas::dot);
  py::def("scal", pycuda::blas::scal);
  py::def("swap", pycuda::blas::swap);
}
