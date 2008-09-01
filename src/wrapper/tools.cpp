#include <vector>
#include "wrap_helpers.hpp"
#include <cuda.hpp>




/* from http://graphics.stanford.edu/~seander/bithacks.html */
static const char log_table_8[] = 
{

  0, 0, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3,
  4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
  5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
  5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
  6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6,
  6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6,
  6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6,
  6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6,
  7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
  7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
  7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
  7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
  7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
  7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
  7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
  7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7
};

static inline unsigned short bitlog2_16(unsigned long v)
{
  if (unsigned long t = v >> 8)
    return 8+log_table_8[t];
  else 
    return log_table_8[v];
}

static inline unsigned short bitlog2_32(unsigned long v)
{
  if (unsigned long t = v >> 16)
    return 16+bitlog2_16(t);
  else 
    return bitlog2_16(v);
}

static inline unsigned short bitlog2(unsigned long v)
{
  if (unsigned long t = v >> 32)
    return 32+bitlog2_32(t);
  else 
    return bitlog2_32(v);
}




namespace
{
  class cuda_allocator
  {
    public:
      typedef CUdeviceptr pointer;
      typedef unsigned long size_type;

      static pointer allocate(size_type s)
      {
        CUdeviceptr devptr;
        CUresult status = cuMemAlloc(&devptr, s);
        if (status == CUDA_SUCCESS)
          return devptr;
        else if (status == CUDA_ERROR_OUT_OF_MEMORY)
          throw std::bad_alloc();
        else 
          throw cuda::error("cuda_allocator::allocate", status);
      }

      static void free(pointer p)
      {
        cuda::mem_free(p);
      }
  };




  template<class Allocator>
  class memory_pool : public cuda::explicit_context_dependent
  {
    public:
      typedef typename Allocator::pointer pointer;
      typedef typename Allocator::size_type size_type;

    private:
      typedef signed short bin_nr;
      static const bin_nr bin_count = 64;
      std::vector<std::vector<pointer> > m_bins;
      Allocator m_allocator;

      // A held block is one that's been released by the application, but that
      // we are keeping around to dish out again.
      unsigned m_held_blocks;

      // An active block is one that is in use by the application.
      unsigned m_active_blocks;

    public:
      memory_pool()
        : m_held_blocks(0), m_active_blocks(0)
      {
        m_bins.resize(bin_count);
      }
      
      ~memory_pool()
      {
        free_held();
      }

    protected:
      bin_nr bin_number(size_type size)
      {
        return bitlog2(size);
      }
      size_type alloc_size(bin_nr bin)
      {
        return (1<<(bin+1)) - 1;
      }

      void inc_held_blocks()
      {
        if (m_held_blocks == 0)
          acquire_context();
        ++m_held_blocks;
      }

      void dec_held_blocks()
      {
        --m_held_blocks;
        if (m_held_blocks == 0)
          release_context();
      }

    public:
      pointer allocate(size_type size)
      {
        bin_nr bin = bin_number(size);
        if (m_bins[bin].size())
        {
          pointer result = m_bins[bin].back();
          m_bins[bin].pop_back();
          dec_held_blocks();
          ++m_active_blocks;
          return result;
        }
        else
        {
          size_type alloc_sz = alloc_size(bin);

          assert(bin_number(alloc_size) == bin);

          bin_nr freeing_in_bin = bin_count-1;

          while (true)
          {
            try
            {
              pointer result = m_allocator.allocate(alloc_sz);
              ++m_active_blocks;

              return result;
            }
            catch (std::bad_alloc)
            {
              // allocation failed, free up some memory

              while (m_bins[freeing_in_bin].size() == 0 && freeing_in_bin >= 0)
                --freeing_in_bin;

              if (freeing_in_bin >= 0)
              {
                m_allocator.free(m_bins[freeing_in_bin].back());
                m_bins[freeing_in_bin].pop_back();
                dec_held_blocks();
              }
              else
                throw;
            }
          }
        }
      }

      void free(pointer p, size_type size)
      {
        --m_active_blocks;
        inc_held_blocks();
        m_bins[bin_number(size)].push_back(p);

        if (m_active_blocks == 0)
        {
          // last deallocation, allow context to go away.
          free_held();
        }
      }

      void free_held()
      {
        for (bin_nr bin = 0; bin < bin_count; ++bin)
        {
          while (m_bins[bin].size())
          {
            m_allocator.free(m_bins[bin].back());
            m_bins[bin].pop_back();
            dec_held_blocks();
          }
        }

        assert(m_held_blocks == 0);
      }

  };




  class pooled_device_allocation 
    : public cuda::context_dependent, public boost::noncopyable
  {
    private:
      typedef memory_pool<cuda_allocator> pool_type;
      boost::shared_ptr<pool_type> m_pool;

      CUdeviceptr m_devptr;
      unsigned long m_size;
      bool m_valid;

    public:
      typedef pool_type::size_type size_type;

      pooled_device_allocation(boost::shared_ptr<pool_type> p, 
          CUdeviceptr devptr, size_type size)
        : m_pool(p), m_devptr(devptr), m_size(size), m_valid(true)
      { 
      }

      void free()
      {
        if (m_valid)
          m_pool->free(m_devptr, m_size);
        else
          throw cuda::error("pooled_device_allocation::free", CUDA_ERROR_INVALID_HANDLE);
      }

      ~pooled_device_allocation()
      {
        if (m_valid)
          m_pool->free(m_devptr, m_size);
      }

      operator CUdeviceptr() const
      { return m_devptr; }
  };




  pooled_device_allocation *pool_allocate(
      boost::shared_ptr<memory_pool<cuda_allocator> > pool,
      memory_pool<cuda_allocator>::size_type sz)
  {
    return new pooled_device_allocation(pool, pool->allocate(sz), sz);
  }




  PyObject *pooled_device_allocation_to_long(pooled_device_allocation const &da)
  {
    return PyLong_FromUnsignedLong((CUdeviceptr) da);
  }
}




void pycuda_expose_tools()
{
  namespace py = boost::python;

  py::def("bitlog2", bitlog2);

  {
    typedef memory_pool<cuda_allocator> cl;
    py::class_<cl, boost::noncopyable, boost::shared_ptr<cl> >("DeviceMemoryPool")
      .DEF_SIMPLE_METHOD(free_held)
      .def("allocate", pool_allocate,
          py::return_value_policy<py::manage_new_object>())
      ;

  }
  {
    typedef pooled_device_allocation cl;
    py::class_<cl, boost::noncopyable>(
        "PooledDeviceAllocation", py::no_init)
      .DEF_SIMPLE_METHOD(free)
      .def("__int__", &cl::operator CUdeviceptr)
      .def("__long__", pooled_device_allocation_to_long)
      ;

    py::implicitly_convertible<pooled_device_allocation, CUdeviceptr>();
  }
}
