// Abstract memory pool implementation
//
// Copyright (C) 2009-17 Andreas Kloeckner
//
// Permission is hereby granted, free of charge, to any person
// obtaining a copy of this software and associated documentation
// files (the "Software"), to deal in the Software without
// restriction, including without limitation the rights to use,
// copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the
// Software is furnished to do so, subject to the following
// conditions:
//
// The above copyright notice and this permission notice shall be
// included in all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
// EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
// OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
// NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
// HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
// WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
// OTHER DEALINGS IN THE SOFTWARE.


#ifndef _AFJDFJSDFSD_PYGPU_HEADER_SEEN_MEMPOOL_HPP
#define _AFJDFJSDFSD_PYGPU_HEADER_SEEN_MEMPOOL_HPP


#include <cassert>
#include <vector>
#include <map>
#include <memory>
#include <ostream>
#include <iostream>
#include "bitlog.hpp"


namespace PYGPU_PACKAGE
{
  // https://stackoverflow.com/a/44175911
  class mp_noncopyable {
  public:
    mp_noncopyable() = default;
    ~mp_noncopyable() = default;

  private:
    mp_noncopyable(const mp_noncopyable&) = delete;
    mp_noncopyable& operator=(const mp_noncopyable&) = delete;
  };

#ifdef PYGPU_PYCUDA
#define PYGPU_SHARED_PTR boost::shared_ptr
#else
#define PYGPU_SHARED_PTR std::shared_ptr
#endif

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




#define always_assert(cond) \
  do { \
    if (!(cond)) \
      throw std::logic_error("mem pool assertion violated: " #cond); \
  } while (false);


  template<class Allocator>
  class memory_pool : mp_noncopyable
  {
    public:
      typedef typename Allocator::pointer_type pointer_type;
      typedef typename Allocator::size_type size_type;

    private:
      typedef uint32_t bin_nr_t;
      typedef std::vector<pointer_type> bin_t;

      typedef std::map<bin_nr_t, bin_t> container_t;
      container_t m_container;
      typedef typename container_t::value_type bin_pair_t;

      std::unique_ptr<Allocator> m_allocator;

      // A held block is one that's been released by the application, but that
      // we are keeping around to dish out again.
      size_type m_held_blocks;

      // An active block is one that is in use by the application.
      size_type m_active_blocks;

      // "Managed" memory is "active" and "held" memory.
      size_type m_managed_bytes;

      // "Active" bytes are bytes under the control of the application.
      // This may be smaller than the actual allocated size reflected
      // in m_managed_bytes.
      size_type m_active_bytes;

      bool m_stop_holding;
      int m_trace;

      unsigned m_leading_bits_in_bin_id;

    public:
      memory_pool(Allocator const &alloc=Allocator(), unsigned leading_bits_in_bin_id=4)
        : m_allocator(alloc.copy()),
        m_held_blocks(0), m_active_blocks(0),
        m_managed_bytes(0), m_active_bytes(0),
        m_stop_holding(false),
        m_trace(false), m_leading_bits_in_bin_id(leading_bits_in_bin_id)
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

    private:
      unsigned mantissa_mask() const
      {
        return (1 << m_leading_bits_in_bin_id) - 1;
      }

    public:
      bin_nr_t bin_number(size_type size)
      {
        signed l = bitlog2(size);
        size_type shifted = signed_right_shift(size, l-signed(m_leading_bits_in_bin_id));
        if (size && (shifted & (1 << m_leading_bits_in_bin_id)) == 0)
          throw std::runtime_error("memory_pool::bin_number: bitlog2 fault");
        size_type chopped = shifted & mantissa_mask();
        return l << m_leading_bits_in_bin_id | chopped;
      }

      void set_trace(bool flag)
      {
        if (flag)
          ++m_trace;
        else
          --m_trace;
      }

      size_type alloc_size(bin_nr_t bin)
      {
        bin_nr_t exponent = bin >> m_leading_bits_in_bin_id;
        bin_nr_t mantissa = bin & mantissa_mask();

        size_type ones = signed_left_shift((size_type) 1,
            signed(exponent)-signed(m_leading_bits_in_bin_id)
            );
        if (ones) ones -= 1;

        size_type head = signed_left_shift(
           (size_type) ((1<<m_leading_bits_in_bin_id) | mantissa),
            signed(exponent)-signed(m_leading_bits_in_bin_id));
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
          auto it_and_inserted = m_container.insert(std::make_pair(bin_nr, bin_t()));
          assert(it_and_inserted.second);
          return it_and_inserted.first->second;
        }
        else
          return it->second;
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

        always_assert(bin_number(alloc_sz) == bin_nr);
        always_assert(alloc_sz >= size);

        if (m_trace)
          std::cout << "[pool] allocation of size " << size << " required new memory" << std::endl;

        try { return get_from_allocator(alloc_sz, size); }
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
          try { return get_from_allocator(alloc_sz, size); }
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
        m_active_bytes -= size;
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
        {
          m_allocator->free(p);
          m_managed_bytes -= alloc_size(bin_nr);
        }
      }

      void free_held()
      {
        for (bin_pair_t &bin_pair: m_container)
        {
          bin_t &bin = bin_pair.second;

          while (bin.size())
          {
            m_allocator->free(bin.back());
            m_managed_bytes -= alloc_size(bin_pair.first);
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

      size_type active_blocks() const
      { return m_active_blocks; }

      size_type held_blocks() const
      { return m_held_blocks; }

      size_type managed_bytes() const
      { return m_managed_bytes; }

      size_type active_bytes() const
      { return m_active_bytes; }

      bool try_to_free_memory()
      {
        // free largest stuff first
        for (typename container_t::reverse_iterator it = m_container.rbegin();
            it != m_container.rend(); ++it)
        {
          bin_pair_t &bin_pair = *it;
          bin_t &bin = bin_pair.second;

          if (bin.size())
          {
            m_allocator->free(bin.back());
            m_managed_bytes -= alloc_size(bin_pair.first);
            bin.pop_back();

            dec_held_blocks();

            return true;
          }
        }

        return false;
      }

    private:
      pointer_type get_from_allocator(size_type alloc_sz, size_type size)
      {
        pointer_type result = m_allocator->allocate(alloc_sz);
        ++m_active_blocks;
        m_managed_bytes += alloc_sz;
        m_active_bytes += size;

        return result;
      }

      pointer_type pop_block_from_bin(bin_t &bin, size_type size)
      {
        pointer_type result = bin.back();
        bin.pop_back();

        dec_held_blocks();
        ++m_active_blocks;
        m_active_bytes += size;

        return result;
      }
  };


  template <class Pool>
  class pooled_allocation : public mp_noncopyable
  {
    public:
      typedef Pool pool_type;
      typedef typename Pool::pointer_type pointer_type;
      typedef typename Pool::size_type size_type;

    private:
      PYGPU_SHARED_PTR<pool_type> m_pool;

      pointer_type m_ptr;
      size_type m_size;
      bool m_valid;

    public:
      pooled_allocation(PYGPU_SHARED_PTR<pool_type> p, size_type size)
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
