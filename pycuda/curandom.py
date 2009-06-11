

md5_code = """
/*
 **********************************************************************
 ** Copyright (C) 1990, RSA Data Security, Inc. All rights reserved. **
 **                                                                  **
 ** License to copy and use this software is granted provided that   **
 ** it is identified as the "RSA Data Security, Inc. MD5 Message     **
 ** Digest Algorithm" in all material mentioning or referencing this **
 ** software or this function.                                       **
 **                                                                  **
 ** License is also granted to make and use derivative works         **
 ** provided that such works are identified as "derived from the RSA **
 ** Data Security, Inc. MD5 Message Digest Algorithm" in all         **
 ** material mentioning or referencing the derived work.             **
 **                                                                  **
 ** RSA Data Security, Inc. makes no representations concerning      **
 ** either the merchantability of this software or the suitability   **
 ** of this software for any particular purpose.  It is provided "as **
 ** is" without express or implied warranty of any kind.             **
 **                                                                  **
 ** These notices must be retained in any copies of any part of this **
 ** documentation and/or software.                                   **
 **********************************************************************
 */

/* F, G and H are basic MD5 functions: selection, majority, parity */
#define F(x, y, z) (((x) & (y)) | ((~x) & (z)))
#define G(x, y, z) (((x) & (z)) | ((y) & (~z)))
#define H(x, y, z) ((x) ^ (y) ^ (z))
#define I(x, y, z) ((y) ^ ((x) | (~z))) 

/* ROTATE_LEFT rotates x left n bits */
#define ROTATE_LEFT(x, n) (((x) << (n)) | ((x) >> (32-(n))))

/* FF, GG, HH, and II transformations for rounds 1, 2, 3, and 4 */
/* Rotation is separate from addition to prevent recomputation */
#define FF(a, b, c, d, x, s, ac) \
  {(a) += F ((b), (c), (d)) + (x) + (ac); \
   (a) = ROTATE_LEFT ((a), (s)); \
   (a) += (b); \
  }
#define GG(a, b, c, d, x, s, ac) \
  {(a) += G ((b), (c), (d)) + (x) + (ac); \
   (a) = ROTATE_LEFT ((a), (s)); \
   (a) += (b); \
  }
#define HH(a, b, c, d, x, s, ac) \
  {(a) += H ((b), (c), (d)) + (x) + (ac); \
   (a) = ROTATE_LEFT ((a), (s)); \
   (a) += (b); \
  }
#define II(a, b, c, d, x, s, ac) \
  {(a) += I ((b), (c), (d)) + (x) + (ac); \
   (a) = ROTATE_LEFT ((a), (s)); \
   (a) += (b); \
  }

#define X0 threadIdx.x
#define X1 threadIdx.y
#define X2 threadIdx.z
#define X3 blockIdx.x
#define X4 blockIdx.y
#define X5 blockIdx.z
#define X6 seed
#define X7 i
#define X8 n
#define X9  blockDim.x
#define X10 blockDim.y
#define X11 blockDim.z
#define X12 gridDim.x
#define X13 gridDim.y
#define X14 gridDim.z
#define X15 0

  unsigned int a = 0x67452301;
  unsigned int b = 0xefcdab89;
  unsigned int c = 0x98badcfe;
  unsigned int d = 0x10325476;
   
  /* Round 1 */
#define S11 7
#define S12 12
#define S13 17
#define S14 22
  FF ( a, b, c, d, X0 , S11, 3614090360); /* 1 */
  FF ( d, a, b, c, X1 , S12, 3905402710); /* 2 */
  FF ( c, d, a, b, X2 , S13,  606105819); /* 3 */
  FF ( b, c, d, a, X3 , S14, 3250441966); /* 4 */
  FF ( a, b, c, d, X4 , S11, 4118548399); /* 5 */
  FF ( d, a, b, c, X5 , S12, 1200080426); /* 6 */
  FF ( c, d, a, b, X6 , S13, 2821735955); /* 7 */
  FF ( b, c, d, a, X7 , S14, 4249261313); /* 8 */
  FF ( a, b, c, d, X8 , S11, 1770035416); /* 9 */
  FF ( d, a, b, c, X9 , S12, 2336552879); /* 10 */
  FF ( c, d, a, b, X10, S13, 4294925233); /* 11 */
  FF ( b, c, d, a, X11, S14, 2304563134); /* 12 */
  FF ( a, b, c, d, X12, S11, 1804603682); /* 13 */
  FF ( d, a, b, c, X13, S12, 4254626195); /* 14 */
  FF ( c, d, a, b, X14, S13, 2792965006); /* 15 */
  FF ( b, c, d, a, X15, S14, 1236535329); /* 16 */

  /* Round 2 */
#define S21 5
#define S22 9
#define S23 14
#define S24 20
  GG ( a, b, c, d, X1 , S21, 4129170786); /* 17 */
  GG ( d, a, b, c, X6 , S22, 3225465664); /* 18 */
  GG ( c, d, a, b, X11, S23,  643717713); /* 19 */
  GG ( b, c, d, a, X0 , S24, 3921069994); /* 20 */
  GG ( a, b, c, d, X5 , S21, 3593408605); /* 21 */
  GG ( d, a, b, c, X10, S22,   38016083); /* 22 */
  GG ( c, d, a, b, X15, S23, 3634488961); /* 23 */
  GG ( b, c, d, a, X4 , S24, 3889429448); /* 24 */
  GG ( a, b, c, d, X9 , S21,  568446438); /* 25 */
  GG ( d, a, b, c, X14, S22, 3275163606); /* 26 */
  GG ( c, d, a, b, X3 , S23, 4107603335); /* 27 */
  GG ( b, c, d, a, X8 , S24, 1163531501); /* 28 */
  GG ( a, b, c, d, X13, S21, 2850285829); /* 29 */
  GG ( d, a, b, c, X2 , S22, 4243563512); /* 30 */
  GG ( c, d, a, b, X7 , S23, 1735328473); /* 31 */
  GG ( b, c, d, a, X12, S24, 2368359562); /* 32 */

  /* Round 3 */
#define S31 4
#define S32 11
#define S33 16
#define S34 23
  HH ( a, b, c, d, X5 , S31, 4294588738); /* 33 */
  HH ( d, a, b, c, X8 , S32, 2272392833); /* 34 */
  HH ( c, d, a, b, X11, S33, 1839030562); /* 35 */
  HH ( b, c, d, a, X14, S34, 4259657740); /* 36 */
  HH ( a, b, c, d, X1 , S31, 2763975236); /* 37 */
  HH ( d, a, b, c, X4 , S32, 1272893353); /* 38 */
  HH ( c, d, a, b, X7 , S33, 4139469664); /* 39 */
  HH ( b, c, d, a, X10, S34, 3200236656); /* 40 */
  HH ( a, b, c, d, X13, S31,  681279174); /* 41 */
  HH ( d, a, b, c, X0 , S32, 3936430074); /* 42 */
  HH ( c, d, a, b, X3 , S33, 3572445317); /* 43 */
  HH ( b, c, d, a, X6 , S34,   76029189); /* 44 */
  HH ( a, b, c, d, X9 , S31, 3654602809); /* 45 */
  HH ( d, a, b, c, X12, S32, 3873151461); /* 46 */
  HH ( c, d, a, b, X15, S33,  530742520); /* 47 */
  HH ( b, c, d, a, X2 , S34, 3299628645); /* 48 */

  /* Round 4 */
#define S41 6
#define S42 10
#define S43 15
#define S44 21
  II ( a, b, c, d, X0 , S41, 4096336452); /* 49 */
  II ( d, a, b, c, X7 , S42, 1126891415); /* 50 */
  II ( c, d, a, b, X14, S43, 2878612391); /* 51 */
  II ( b, c, d, a, X5 , S44, 4237533241); /* 52 */
  II ( a, b, c, d, X12, S41, 1700485571); /* 53 */
  II ( d, a, b, c, X3 , S42, 2399980690); /* 54 */
  II ( c, d, a, b, X10, S43, 4293915773); /* 55 */
  II ( b, c, d, a, X1 , S44, 2240044497); /* 56 */
  II ( a, b, c, d, X8 , S41, 1873313359); /* 57 */
  II ( d, a, b, c, X15, S42, 4264355552); /* 58 */
  II ( c, d, a, b, X6 , S43, 2734768916); /* 59 */
  II ( b, c, d, a, X13, S44, 1309151649); /* 60 */
  II ( a, b, c, d, X4 , S41, 4149444226); /* 61 */
  II ( d, a, b, c, X11, S42, 3174756917); /* 62 */
  II ( c, d, a, b, X2 , S43,  718787259); /* 63 */
  II ( b, c, d, a, X9 , S44, 3951481745); /* 64 */

  a += 0x67452301;
  b += 0xefcdab89;
  c += 0x98badcfe;
  d += 0x10325476;
"""

import numpy

def rand(shape, dtype=numpy.float32, stream=None):
    from pycuda.gpuarray import GPUArray
    from pycuda.elementwise import get_elwise_kernel

    result = GPUArray(shape, dtype)
    
    if dtype == numpy.float32:
        func = get_elwise_kernel(
            "float *dest, unsigned int seed", 
            md5_code + """
            #define POW_2_M32 (1/4294967296.0f)
            dest[i] = a*POW_2_M32;
            if ((i += total_threads) < n)
                dest[i] = b*POW_2_M32;
            if ((i += total_threads) < n)
                dest[i] = c*POW_2_M32;
            if ((i += total_threads) < n)
                dest[i] = d*POW_2_M32;
            """,
            "md5_rng_float")
    elif dtype == numpy.float64:
        func = get_elwise_kernel(
            "double *dest, unsigned int seed", 
            md5_code + """
            #define POW_2_M32 (1/4294967296.0)
            #define POW_2_M64 (1/18446744073709551616.)

            dest[i] = a*POW_2_M32 + b*POW_2_M64;

            if ((i += total_threads) < n)
            {
              dest[i] = c*POW_2_M32 + d*POW_2_M64;
            }
            """,
            "md5_rng_float")
    elif dtype in [numpy.int32, numpy.uint32]:
        func = get_elwise_kernel(
            "unsigned int *dest, unsigned int seed", 
            md5_code + """
            dest[i] = a;
            if ((i += total_threads) < n)
                dest[i] = b;
            if ((i += total_threads) < n)
                dest[i] = c;
            if ((i += total_threads) < n)
                dest[i] = d;
            """,
            "md5_rng_int")
    else:
        raise NotImplementedError;

    func.set_block_shape(*result._block)
    func.prepared_async_call(result._grid, stream,
            result.gpudata, numpy.random.randint(2**31-1), result.size)
    
    return result


if __name__ == "__main__":
    import sys, pycuda.autoinit

    if "generate" in sys.argv[1:]:
        N = 256
        print N, "MB"
        r = rand((N*2**18,), numpy.uint32)
        print "generated"
        r.get().tofile("random.dat")
        print "written"

    else: 
        from pylab import plot, show, subplot
        N = 250
        r1 = rand((N,), numpy.uint32)
        r2 = rand((N,), numpy.int32)
        r3 = rand((N,), numpy.float32)
    
        subplot(131); plot( r1.get(),"x-")
        subplot(132); plot( r2.get(),"x-")
        subplot(133); plot( r3.get(),"x-")
        show()
