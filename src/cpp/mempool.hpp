// Abstract memory pool implementation




#ifndef _AFJDFJSDFSD_PYGPU_HEADER_SEEN_MEMPOOL_HPP
#define _AFJDFJSDFSD_PYGPU_HEADER_SEEN_MEMPOOL_HPP




#include <boost/ptr_container/ptr_map.hpp>
#include <boost/foreach.hpp>
#include <boost/format.hpp>
#include "bitlog.hpp"




namespace PYGPU_PACKAGE
{
  template <class T>
  inline T signed_left_shift(T x, signed shift_amount)
  {
    if (shift_amount < 0)
      return x >> -shift_amount;
    else
      return x << shift_amount;
  }




  template <class T>
  inline T signed_right_shift(T x, signed shift_amount)
  {
    if (shift_amount < 0)
      return x << -shift_amount;
    else
      return x >> shift_amount;
  }




  template<class Allocator>
  class memory_pool
  {
    public:
      typedef typename Allocator::pointer_type pointer_type;
      typedef typename Allocator::size_type size_type;

    private:
      typedef boost::uint32_t bin_nr_t;
      typedef std::vector<pointer_type> bin_t;

      typedef boost::ptr_map<bin_nr_t, bin_t > container_t;
      container_t m_container;
      typedef typename container_t::value_type bin_pair_t;

      std::auto_ptr<Allocator> m_allocator;

      // A held block is one that's been released by the application, but that
      // we are keeping around to dish out again.
      unsigned m_held_blocks;

      // An active block is one that is in use by the application.
      unsigned m_active_blocks;

      bool m_stop_holding;
      int m_trace;

    public:
      memory_pool(Allocator const &alloc=Allocator())
        : m_allocator(alloc.copy()),
        m_held_blocks(0), m_active_blocks(0), m_stop_holding(false),
        m_trace(false)
      {
        if (m_allocator->is_deferred())
        {
          PyErr_WarnEx(PyExc_UserWarning, "Memory pools expect non-deferred "
              "semantics from their allocators. You passed a deferred "
              "allocator, i.e. an allocator whose allocations can turn out to "
              "be unavailable long after allocation.", 1);
        }
      }

      virtual ~memory_pool()
      { free_held(); }

      static const unsigned mantissa_bits = 2;
      static const unsigned mantissa_mask = (1 << mantissa_bits) - 1;

      static bin_nr_t bin_number(size_type size)
      {
        signed l = bitlog2(size);
        size_type shifted = signed_right_shift(size, l-signed(mantissa_bits));
        if (size && (shifted & (1 << mantissa_bits)) == 0)
          throw std::runtime_error("memory_pool::bin_number: bitlog2 fault");
        size_type chopped = shifted & mantissa_mask;
        return l << mantissa_bits | chopped;
      }

      void set_trace(bool flag)
      {
        if (flag)
          ++m_trace;
        else
          --m_trace;
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
          start_holding_blocks();
        ++m_held_blocks;
      }

      void dec_held_blocks()
      {
        --m_held_blocks;
        if (m_held_blocks == 0)
          stop_holding_blocks();
      }

      virtual void start_holding_blocks()
      { }

      virtual void stop_holding_blocks()
      { }

    public:
      pointer_type allocate(size_type size)
      {
        bin_nr_t bin_nr = bin_number(size);
        bin_t &bin = get_bin(bin_nr);

        if (bin.size())
        {
          if (m_trace)
            std::cout
              << "[pool] allocation of size " << size << " served from bin " << bin_nr
              << " which contained " << bin.size() << " entries" << std::endl;
          return pop_block_from_bin(bin, size);
        }

        size_type alloc_sz = alloc_size(bin_nr);

        assert(bin_number(alloc_sz) == bin_nr);

        if (m_trace)
          std::cout << "[pool] allocation of size " << size << " required new memory" << std::endl;

        try { return get_from_allocator(alloc_sz); }
        catch (PYGPU_PACKAGE::error &e)
        {
          if (!e.is_out_of_memory())
            throw;
        }

        if (m_trace)
          std::cout << "[pool] allocation triggered OOM, running GC" << std::endl;

        m_allocator->try_release_blocks();
        if (bin.size())
          return pop_block_from_bin(bin, size);

        if (m_trace)
          std::cout << "[pool] allocation still OOM after GC" << std::endl;

        while (try_to_free_memory())
        {
          try { return get_from_allocator(alloc_sz); }
          catch (PYGPU_PACKAGE::error &e)
          {
            if (!e.is_out_of_memory())
              throw;
          }
        }

        throw PYGPU_PACKAGE::error(
            "memory_pool::allocate",
#ifdef PYGPU_PYCUDA
            CUDA_ERROR_OUT_OF_MEMORY,
#endif
#ifdef PYGPU_PYOPENCL
            CL_MEM_OBJECT_ALLOCATION_FAILURE,
#endif
            "failed to free memory for allocation");
      }

      void free(pointer_type p, size_type size)
      {
        --m_active_blocks;
        bin_nr_t bin_nr = bin_number(size);

        if (!m_stop_holding)
        {
          inc_held_blocks();
          get_bin(bin_nr).push_back(p);

          if (m_trace)
            std::cout << "[pool] block of size " << size << " returned to bin "
              << bin_nr << " which now contains " << get_bin(bin_nr).size()
              << " entries" << std::endl;
        }
        else
          m_allocator->free(p);
      }

      void free_held()
      {
        BOOST_FOREACH(bin_pair_t bin_pair, m_container)
        {
          bin_t &bin = *bin_pair.second;

          while (bin.size())
          {
            m_allocator->free(bin.back());
            bin.pop_back();

            dec_held_blocks();
          }
        }

        assert(m_held_blocks == 0);
      }

      void stop_holding()
      {
        m_stop_holding = true;
        free_held();
      }

      unsigned active_blocks()
      { return m_active_blocks; }

      unsigned held_blocks()
      { return m_held_blocks; }

      bool try_to_free_memory()
      {
        BOOST_FOREACH(bin_pair_t bin_pair,
            // free largest stuff first
            std::make_pair(m_container.rbegin(), m_container.rend()))
        {
          bin_t &bin = *bin_pair.second;

          if (bin.size())
          {
            m_allocator->free(bin.back());
            bin.pop_back();

            dec_held_blocks();

            return true;
          }
        }

        return false;
      }

    private:
      pointer_type get_from_allocator(size_type alloc_sz)
      {
        pointer_type result = m_allocator->allocate(alloc_sz);
        ++m_active_blocks;

        return result;
      }

      pointer_type pop_block_from_bin(bin_t &bin, size_type size)
      {
        pointer_type result = bin.back();
        bin.pop_back();

        dec_held_blocks();
        ++m_active_blocks;

        return result;
      }
  };





  template <class Pool>
  class pooled_allocation : public boost::noncopyable
  {
    public:
      typedef Pool pool_type;
      typedef typename Pool::pointer_type pointer_type;
      typedef typename Pool::size_type size_type;

    private:
      boost::shared_ptr<pool_type> m_pool;

      pointer_type m_ptr;
      size_type m_size;
      bool m_valid;

    public:
      pooled_allocation(boost::shared_ptr<pool_type> p, size_type size)
        : m_pool(p), m_ptr(p->allocate(size)), m_size(size), m_valid(true)
      { }

      ~pooled_allocation()
      {
        if (m_valid)
          free();
      }

      void free()
      {
        if (m_valid)
        {
          m_pool->free(m_ptr, m_size);
          m_valid = false;
        }
        else
          throw PYGPU_PACKAGE::error(
              "pooled_device_allocation::free", 
#ifdef PYGPU_PYCUDA
              CUDA_ERROR_INVALID_HANDLE
#endif
#ifdef PYGPU_PYOPENCL
              CL_INVALID_VALUE
#endif
              );
      }

      pointer_type ptr() const
      { return m_ptr; }

      size_type size() const
      { return m_size; }
  };
}




#endif
