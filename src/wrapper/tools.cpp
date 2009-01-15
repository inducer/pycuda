#include <vector>
#include "wrap_helpers.hpp"
#include <cuda.hpp>
#include <boost/ptr_container/ptr_map.hpp>
#include <boost/cstdint.hpp>
#include <boost/foreach.hpp>
#include <climits>




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

static inline unsigned bitlog2_16(boost::uint16_t v)
{
  if (unsigned long t = v >> 8)
    return 8+log_table_8[t];
  else 
    return log_table_8[v];
}

static inline unsigned bitlog2_32(boost::uint32_t v)
{
  if (uint16_t t = v >> 16)
    return 16+bitlog2_16(t);
  else 
    return bitlog2_16(v);
}

static inline unsigned bitlog2(unsigned long v)
{
#if (ULONG_MAX != 4294967295)
  if (boost::uint32_t t = v >> 32)
    return 32+bitlog2_32(t);
  else 
#endif
    return bitlog2_32(v);
}




namespace
{
  class cuda_allocator : public cuda::context_dependent
  {
    public:
      typedef CUdeviceptr pointer;
      typedef unsigned long size_type;

      pointer allocate(size_type s)
      {
        cuda::scoped_context_activation ca(get_context());

        CUdeviceptr devptr;
        CUresult status = cuMemAlloc(&devptr, s);
        if (status == CUDA_SUCCESS)
          return devptr;
        else if (status == CUDA_ERROR_OUT_OF_MEMORY)
          throw std::bad_alloc();
        else 
          throw cuda::error("cuda_allocator::allocate", status);
      }

      void free(pointer p)
      {
        cuda::scoped_context_activation ca(get_context());
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
      typedef boost::uint32_t bin_nr_t;
      typedef std::vector<pointer> bin_t;

      typedef boost::ptr_map<bin_nr_t, bin_t > container_t;
      container_t m_container;
      typedef typename container_t::value_type bin_pair_t;

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
      }
      
      ~memory_pool()
      {
        free_held();
      }

      static const unsigned mantissa_bits = 2;
      static const unsigned mantissa_mask = (1 << mantissa_bits) - 1;

      static size_type signed_left_shift(size_type x, signed shift_amount)
      {
        if (shift_amount < 0)
          return x >> -shift_amount;
        else
          return x << shift_amount;
      }

      static size_type signed_right_shift(size_type x, signed shift_amount)
      {
        if (shift_amount < 0)
          return x << -shift_amount;
        else
          return x >> shift_amount;
      }

      static bin_nr_t bin_number(size_type size)
      {
        signed l = bitlog2(size);
        size_type shifted = signed_right_shift(size, l-signed(mantissa_bits));
        if (size && (shifted & (1 << mantissa_bits)) == 0)
          throw std::runtime_error("memory_pool::bin_number: bitlog2 fault");
        size_type chopped = shifted & mantissa_mask;
        return l << mantissa_bits | chopped;
      }

      static size_type alloc_size(bin_nr_t bin)
      {
        bin_nr_t exponent = bin >> mantissa_bits;
        bin_nr_t mantissa = bin & mantissa_mask;

        size_type ones = signed_left_shift(1, 
            signed(exponent)-signed(mantissa_bits)
            );
        if (ones) ones -= 1;

        size_type head = signed_left_shift(
           (1<<mantissa_bits) | mantissa, 
            signed(exponent)-signed(mantissa_bits));
        if (ones & head)
          throw std::runtime_error("memory_pool::alloc_size: bit-counting fault");
        return head | ones;
      }

    protected:
      bin_t &get_bin(bin_nr_t bin_nr)
      {
        typename container_t::iterator it = m_container.find(bin_nr);
        if (it == m_container.end())
        {
          bin_t *new_bin = new bin_t;
          m_container.insert(bin_nr, new_bin);
          return *new_bin;
        }
        else
          return *it->second;
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
        bin_nr_t bin_nr = bin_number(size);
        bin_t &bin = get_bin(bin_nr);
        
        if (bin.size())
        {
          pointer result = bin.back();
          bin.pop_back();
          dec_held_blocks();
          ++m_active_blocks;
          return result;
        }
        else
        {
          size_type alloc_sz = alloc_size(bin_nr);

          assert(bin_number(alloc_sz) == bin);

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

              bool freed_some = false;
              BOOST_FOREACH(bin_pair_t bin_pair, 
                  // free largest stuff first
                  std::make_pair(m_container.rbegin(), m_container.rend()))
              {
                bin_t &bin = *bin_pair.second;

                if (bin.size())
                {
                  m_allocator.free(bin.back());
                  bin.pop_back();
                  dec_held_blocks();
                  freed_some = true;
                  break;
                }
              }

              if (!freed_some)
                throw;
            }
          }
        }
      }

      void free(pointer p, size_type size)
      {
        --m_active_blocks;
        inc_held_blocks();
        get_bin(bin_number(size)).push_back(p);

        if (m_active_blocks == 0)
        {
          // last deallocation, allow context to go away.
          free_held();
        }
      }

      void free_held()
      {
        BOOST_FOREACH(bin_pair_t bin_pair, m_container)
        {
          bin_t &bin = *bin_pair.second;

          while (bin.size())
          {
            m_allocator.free(bin.back());
            bin.pop_back();
            dec_held_blocks();
          }
        }

        assert(m_held_blocks == 0);
      }

      unsigned active_blocks()
      { return m_active_blocks; }

      unsigned held_blocks()
      { return m_held_blocks; }

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

      unsigned long size() const
      { return m_size; }
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
      .add_property("held_blocks", &cl::held_blocks)
      .add_property("active_blocks", &cl::active_blocks)
      .DEF_SIMPLE_METHOD(bin_number)
      .DEF_SIMPLE_METHOD(alloc_size)
      .staticmethod("bin_number")
      .staticmethod("alloc_size")
      ;
  }
  {
    typedef pooled_device_allocation cl;
    py::class_<cl, boost::noncopyable>(
        "PooledDeviceAllocation", py::no_init)
      .DEF_SIMPLE_METHOD(free)
      .def("__int__", &cl::operator CUdeviceptr)
      .def("__long__", pooled_device_allocation_to_long)
      .def("__len__", &cl::size)
      ;

    py::implicitly_convertible<pooled_device_allocation, CUdeviceptr>();
  }
}
