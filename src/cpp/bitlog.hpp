// Base-2 logarithm bithack.
//
// Copyright (C) 2009 Andreas Kloeckner
// Copyright (C) Sean Eron Anderson (in the public domain)
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


#ifndef _AFJDFJSDFSD_PYOPENCL_HEADER_SEEN_BITLOG_HPP
#define _AFJDFJSDFSD_PYOPENCL_HEADER_SEEN_BITLOG_HPP


#include <climits>
#include <cstdint>


namespace pycuda
{
  /* from http://graphics.stanford.edu/~seander/bithacks.html */

  extern const char log_table_8[];

  inline unsigned bitlog2_16(uint16_t v)
  {
    if (unsigned long t = v >> 8)
      return 8+log_table_8[t];
    else
      return log_table_8[v];
  }

  inline unsigned bitlog2_32(uint32_t v)
  {
    if (uint16_t t = v >> 16)
      return 16+bitlog2_16(t);
    else
      return bitlog2_16(v);
  }

#if defined(UINT64_MAX)
  inline unsigned bitlog2(uint64_t v)
  {
    if (uint32_t t = v >> 32)
      return 32+bitlog2_32(t);
    else
      return bitlog2_32(v);
  }
#else
  inline unsigned bitlog2(unsigned long v)
  {
#if (ULONG_MAX != 4294967295)
    if (uint32_t t = v >> 32)
      return 32+bitlog2_32(t);
    else
#endif
      return bitlog2_32(v);
  }
#endif
}





#endif
