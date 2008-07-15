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
          throw cuda::error("mem_pool_alloc", status);
      }

      static void free(pointer p)
      {
        cuda::mem_free(p);
      }
  };




  template<class Allocator>
  class memory_pool
  {
    public:
      typedef typename Allocator::pointer pointer;
      typedef typename Allocator::size_type size_type;

    private:
      typedef signed short bin_nr;
      static const bin_nr bin_count = 64;
      std::vector<std::vector<pointer> > m_bins;
      Allocator m_allocator;

    public:
      memory_pool()
      {
        m_bins.resize(bin_count);
      }

      pointer allocate(size_type size)
      {
        bin_nr bin = bitlog2(size);
        if (m_bins[bin].size())
        {
          pointer result = m_bins[bin].back();
          m_bins[bin].pop_back();
          return result;
        }
        else
        {
          size_type alloc_sz = 1<<bin;
          bin_nr freeing_in_bin = bin_count-1;
          while (true)
          {
            try
            {
              return m_allocator.allocate(alloc_sz);
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
              }
              else
                throw;
            }
          }
        }
      }

      void free(pointer p, size_type size)
      {
        m_bins[bitlog2(size)].push_back(p);
      }

      void free_all()
      {
        for (bin_nr bin = 0; bin < bin_count; ++bin)
        {
          while (m_bins[bin].size())
          {
            m_allocator.free(m_bins[bin].back());
            m_bins[bin].pop_back();
          }
        }
      }

  };




  class pooled_device_allocation 
    : public cuda::context_dependent, public boost::noncopyable
  {
    private:
      CUdeviceptr m_devptr;
      unsigned long m_size;
      bool m_valid;
      typedef memory_pool<cuda_allocator> pool_type;

    public:
      static pool_type m_pool;

      typedef pool_type::size_type size_type;

      pooled_device_allocation(size_type size)
        : m_devptr(m_pool.allocate(size)), m_size(size), m_valid(true)
      { 
      }

      void free()
      {
        if (m_valid)
          m_pool.free(m_devptr, m_size);
        else
          throw cuda::error("pooled_device_allocation::free", CUDA_ERROR_INVALID_HANDLE);
      }

      ~pooled_device_allocation()
      {
        if (m_valid)
          m_pool.free(m_devptr, m_size);
      }

      operator CUdeviceptr()
      { return m_devptr; }
  };




  memory_pool<cuda_allocator> pooled_device_allocation::m_pool;
}




void pycuda_expose_tools()
{
  namespace py = boost::python;

  py::def("bitlog2", bitlog2);

  {
    typedef memory_pool<cuda_allocator> cl;
    py::class_<cl, boost::noncopyable>("MemoryPool", py::no_init)
      .DEF_SIMPLE_METHOD(free_all)
      ;

  }
  {
    typedef pooled_device_allocation cl;
    py::class_<cl, boost::noncopyable>(
        "PooledDeviceAllocation", py::init<cl::size_type>())
      .DEF_SIMPLE_RO_MEMBER(pool)
      .DEF_SIMPLE_METHOD(free)
      ;

    py::implicitly_convertible<pooled_device_allocation, CUdeviceptr>();
  }
}
