/*
 * Copyright (c) 1999
 * Silicon Graphics Computer Systems, Inc.
 *
 * Copyright (c) 1999
 * Boris Fomitchev
 *
 * This material is provided "as is", with absolutely no warranty expressed
 * or implied. Any use is at your own risk.
 *
 * Permission to use or copy this software for any purpose is hereby granted
 * without fee, provided the above notices are retained on all copies.
 * Permission to modify the code and to distribute modified code is granted,
 * provided the above notices are retained, and a notice that the code was
 * modified is included with the above copyright notice.
 *
 * Adapted for PyCUDA by Andreas Kloeckner 2009.
 */
#ifndef PYCUDA_COMPLEX_HPP_SEEN
#define PYCUDA_COMPLEX_HPP_SEEN



extern "C++" {
namespace pycuda {

#define _STLP_USE_NO_IOSTREAMS
#define _STLP_DECLSPEC /* empty */
#define _STLP_CLASS_DECLSPEC /* empty */
#define _STLP_FUNCTION_TMPL_PARTIAL_ORDER
#define _STLP_TEMPLATE_NULL template<>

template <class _Tp>
struct complex {
  typedef _Tp value_type;
  typedef complex<_Tp> _Self;

  // Constructors, destructor, assignment operator.
  __device__
  complex() : _M_re(0), _M_im(0) {}
  __device__
  complex(const value_type& __x)
    : _M_re(__x), _M_im(0) {}
  __device__
  complex(const value_type& __x, const value_type& __y)
    : _M_re(__x), _M_im(__y) {}
  __device__
  complex(const _Self& __z)
    : _M_re(__z._M_re), _M_im(__z._M_im) {}

  __device__
  _Self& operator=(const _Self& __z) {
    _M_re = __z._M_re;
    _M_im = __z._M_im;
    return *this;
  }
  __device__
  volatile _Self& operator=(const _Self& __z) volatile {
    _M_re = __z._M_re;
    _M_im = __z._M_im;
    return *this;
  }


  template <class _Tp2>
  __device__
  explicit complex(const complex<_Tp2>& __z)
    : _M_re(__z._M_re), _M_im(__z._M_im) {}

  template <class _Tp2>
  __device__
  _Self& operator=(const complex<_Tp2>& __z) {
    _M_re = __z._M_re;
    _M_im = __z._M_im;
    return *this;
  }
  template <class _Tp2>
  __device__
  volatile _Self& operator=(const complex<_Tp2>& __z) volatile {
    _M_re = __z._M_re;
    _M_im = __z._M_im;
    return *this;
  }

  // Element access.
  __device__
  value_type real() const { return _M_re; }
  __device__
  value_type imag() const { return _M_im; }
  __device__
  void real(value_type val) { _M_re = val; }
  __device__
  void imag(value_type val) { _M_im = val; }

  // Arithmetic op= operations involving one real argument.

  __device__
  _Self& operator= (const value_type& __x) {
    _M_re = __x;
    _M_im = 0;
    return *this;
  }
  __device__
  volatile _Self& operator= (const value_type& __x) volatile {
    _M_re = __x;
    _M_im = 0;
    return *this;
  }
  __device__
  _Self& operator+= (const value_type& __x) {
    _M_re += __x;
    return *this;
  }
  __device__
  _Self& operator-= (const value_type& __x) {
    _M_re -= __x;
    return *this;
  }
  __device__
  _Self& operator*= (const value_type& __x) {
    _M_re *= __x;
    _M_im *= __x;
    return *this;
  }
  __device__
  _Self& operator/= (const value_type& __x) {
    _M_re /= __x;
    _M_im /= __x;
    return *this;
  }

  // Arithmetic op= operations involving two complex arguments.

  static void  __device__ _div(const value_type& __z1_r, const value_type& __z1_i,
                               const value_type& __z2_r, const value_type& __z2_i,
                               value_type& __res_r, value_type& __res_i);

  static void __device__ _div(const value_type& __z1_r,
                              const value_type& __z2_r, const value_type& __z2_i,
                              value_type& __res_r, value_type& __res_i);

  template <class _Tp2> __device__ _Self& operator+= (const complex<_Tp2>& __z) {
    _M_re += __z._M_re;
    _M_im += __z._M_im;
    return *this;
  }

  template <class _Tp2> __device__ _Self& operator-= (const complex<_Tp2>& __z) {
    _M_re -= __z._M_re;
    _M_im -= __z._M_im;
    return *this;
  }

  template <class _Tp2> __device__ _Self& operator*= (const complex<_Tp2>& __z) {
    value_type __r = _M_re * __z._M_re - _M_im * __z._M_im;
    value_type __i = _M_re * __z._M_im + _M_im * __z._M_re;
    _M_re = __r;
    _M_im = __i;
    return *this;
  }

  template <class _Tp2> __device__ _Self& operator/= (const complex<_Tp2>& __z) {
    value_type __r;
    value_type __i;
    _div(_M_re, _M_im, __z._M_re, __z._M_im, __r, __i);
    _M_re = __r;
    _M_im = __i;
    return *this;
  }

  __device__
  _Self& operator+= (const _Self& __z) {
    _M_re += __z._M_re;
    _M_im += __z._M_im;
    return *this;
  }

  __device__
  _Self& operator-= (const _Self& __z) {
    _M_re -= __z._M_re;
    _M_im -= __z._M_im;
    return *this;
  }

  __device__
  _Self& operator*= (const _Self& __z) {
    value_type __r = _M_re * __z._M_re - _M_im * __z._M_im;
    value_type __i = _M_re * __z._M_im + _M_im * __z._M_re;
    _M_re = __r;
    _M_im = __i;
    return *this;
  }

  __device__
  _Self& operator/= (const _Self& __z) {
    value_type __r;
    value_type __i;
    _div(_M_re, _M_im, __z._M_re, __z._M_im, __r, __i);
    _M_re = __r;
    _M_im = __i;
    return *this;
  }

  // Data members.
  value_type _M_re;
  value_type _M_im;
};

// Explicit specializations for float, double, long double.  The only
// reason for these specializations is to enable automatic conversions
// from complex<float> to complex<double>, and complex<double> to
// complex<long double>.

_STLP_TEMPLATE_NULL
struct _STLP_CLASS_DECLSPEC complex<float> {
  typedef float value_type;
  typedef complex<float> _Self;
  // Constructors, destructor, assignment operator.

  __device__ complex(value_type __x = 0.0f, value_type __y = 0.0f)
    : _M_re(__x), _M_im(__y) {}

  __device__ complex(const complex<float>& __z) 
    : _M_re(__z._M_re), _M_im(__z._M_im) {}

  inline explicit __device__ complex(const complex<double>& __z);
  // Element access.
  value_type __device__ real() const { return _M_re; }
  value_type __device__ imag() const { return _M_im; }
  void __device__ real(value_type val) { _M_re = val; }
  void __device__ imag(value_type val) { _M_im = val; }

  // Arithmetic op= operations involving one real argument.

  __device__ _Self& operator= (value_type __x) {
    _M_re = __x;
    _M_im = 0.0f;
    return *this;
  }
  volatile __device__ _Self& operator= (value_type __x) volatile {
    _M_re = __x;
    _M_im = 0.0f;
    return *this;
  }
  __device__ _Self& operator+= (value_type __x) {
    _M_re += __x;
    return *this;
  }
  __device__ _Self& operator-= (value_type __x) {
    _M_re -= __x;
    return *this;
  }
  __device__ _Self& operator*= (value_type __x) {
    _M_re *= __x;
    _M_im *= __x;
    return *this;
  }
  __device__ _Self& operator/= (value_type __x) {
    _M_re /= __x;
    _M_im /= __x;
    return *this;
  }

  // Arithmetic op= operations involving two complex arguments.

  static __device__ void _div(const float& __z1_r, const float& __z1_i,
                              const float& __z2_r, const float& __z2_i,
                              float& __res_r, float& __res_i);

  static __device__ void _div(const float& __z1_r,
                              const float& __z2_r, const float& __z2_i,
                              float& __res_r, float& __res_i);

  template <class _Tp2>
  __device__ 
  complex<float>& operator=(const complex<_Tp2>& __z) {
    _M_re = __z._M_re;
    _M_im = __z._M_im;
    return *this;
  }

  template <class _Tp2>
  __device__ 
  volatile complex<float>& operator=(const complex<_Tp2>& __z) volatile {
    _M_re = __z._M_re;
    _M_im = __z._M_im;
    return *this;
  }


  template <class _Tp2>
  __device__ 
  complex<float>& operator+= (const complex<_Tp2>& __z) {
    _M_re += __z._M_re;
    _M_im += __z._M_im;
    return *this;
  }

  template <class _Tp2>
  __device__ 
  complex<float>& operator-= (const complex<_Tp2>& __z) {
    _M_re -= __z._M_re;
    _M_im -= __z._M_im;
    return *this;
  }

  template <class _Tp2>
  __device__ 
  complex<float>& operator*= (const complex<_Tp2>& __z) {
    float __r = _M_re * __z._M_re - _M_im * __z._M_im;
    float __i = _M_re * __z._M_im + _M_im * __z._M_re;
    _M_re = __r;
    _M_im = __i;
    return *this;
  }

  template <class _Tp2>
  __device__ 
  complex<float>& operator/= (const complex<_Tp2>& __z) {
    float __r;
    float __i;
    _div(_M_re, _M_im, __z._M_re, __z._M_im, __r, __i);
    _M_re = __r;
    _M_im = __i;
    return *this;
  }

  __device__ 
  _Self& operator=(const _Self& __z) {
    _M_re = __z._M_re;
    _M_im = __z._M_im;
    return *this;
  }

  __device__ 
  volatile _Self& operator=(const _Self& __z) volatile {
    _M_re = __z._M_re;
    _M_im = __z._M_im;
    return *this;
  }


  __device__ 
  _Self& operator+= (const _Self& __z) {
    _M_re += __z._M_re;
    _M_im += __z._M_im;
    return *this;
  }

  __device__ 
  _Self& operator-= (const _Self& __z) {
    _M_re -= __z._M_re;
    _M_im -= __z._M_im;
    return *this;
  }

  __device__ 
  _Self& operator*= (const _Self& __z) {
    value_type __r = _M_re * __z._M_re - _M_im * __z._M_im;
    value_type __i = _M_re * __z._M_im + _M_im * __z._M_re;
    _M_re = __r;
    _M_im = __i;
    return *this;
  }

  __device__ 
  _Self& operator/= (const _Self& __z) {
    value_type __r;
    value_type __i;
    _div(_M_re, _M_im, __z._M_re, __z._M_im, __r, __i);
    _M_re = __r;
    _M_im = __i;
    return *this;
  }

  // Data members.
  value_type _M_re;
  value_type _M_im;
};

template<>
struct _STLP_CLASS_DECLSPEC complex<double> {
  typedef double value_type;
  typedef complex<double> _Self;

  // Constructors, destructor, assignment operator.

  __device__
  complex(value_type __x = 0.0, value_type __y = 0.0)
    : _M_re(__x), _M_im(__y) {}

  __device__
  complex(const complex<double>& __z)
    : _M_re(__z._M_re), _M_im(__z._M_im) {}
  __device__
  inline complex(const complex<float>& __z);
  // Element access.
  __device__
  value_type real() const { return _M_re; }
  __device__
  value_type imag() const { return _M_im; }
  __device__
  void real(value_type val) { _M_re = val; }
  __device__
  void imag(value_type val) { _M_im = val; }

  // Arithmetic op= operations involving one real argument.

  __device__
  _Self& operator= (value_type __x) {
    _M_re = __x;
    _M_im = 0.0;
    return *this;
  }
  __device__
  volatile _Self& operator= (value_type __x) volatile {
    _M_re = __x;
    _M_im = 0.0;
    return *this;
  }
  __device__
  _Self& operator+= (value_type __x) {
    _M_re += __x;
    return *this;
  }
  __device__
  _Self& operator-= (value_type __x) {
    _M_re -= __x;
    return *this;
  }
  __device__
  _Self& operator*= (value_type __x) {
    _M_re *= __x;
    _M_im *= __x;
    return *this;
  }
  __device__
  _Self& operator/= (value_type __x) {
    _M_re /= __x;
    _M_im /= __x;
    return *this;
  }

  // Arithmetic op= operations involving two complex arguments.

  static __device__ void _div(const double& __z1_r, const double& __z1_i,
                              const double& __z2_r, const double& __z2_i,
                              double& __res_r, double& __res_i);
  static __device__ void _div(const double& __z1_r,
                              const double& __z2_r, const double& __z2_i,
                              double& __res_r, double& __res_i);

#if defined (_STLP_FUNCTION_TMPL_PARTIAL_ORDER)
  template <class _Tp2>
  __device__
  complex<double>& operator=(const complex<_Tp2>& __z) {
    _M_re = __z._M_re;
    _M_im = __z._M_im;
    return *this;
  }

  template <class _Tp2>
  __device__
  volatile complex<double>& operator=(const volatile complex<_Tp2>& __z) {
    _M_re = __z._M_re;
    _M_im = __z._M_im;
    return *this;
  }

  template <class _Tp2>
  __device__
  complex<double>& operator+= (const complex<_Tp2>& __z) {
    _M_re += __z._M_re;
    _M_im += __z._M_im;
    return *this;
  }

  template <class _Tp2>
  __device__ 
  complex<double>& operator-= (const complex<_Tp2>& __z) {
    _M_re -= __z._M_re;
    _M_im -= __z._M_im;
    return *this;
  }

  template <class _Tp2>
  __device__ 
  complex<double>& operator*= (const complex<_Tp2>& __z) {
    double __r = _M_re * __z._M_re - _M_im * __z._M_im;
    double __i = _M_re * __z._M_im + _M_im * __z._M_re;
    _M_re = __r;
    _M_im = __i;
    return *this;
  }

  template <class _Tp2>
  __device__ 
  complex<double>& operator/= (const complex<_Tp2>& __z) {
    double __r;
    double __i;
    _div(_M_re, _M_im, __z._M_re, __z._M_im, __r, __i);
    _M_re = __r;
    _M_im = __i;
    return *this;
  }

#endif /* _STLP_FUNCTION_TMPL_PARTIAL_ORDER */
  __device__ 
  _Self& operator=(const _Self& __z) {
    _M_re = __z._M_re;
    _M_im = __z._M_im;
    return *this;
  }

  __device__ 
  volatile _Self& operator=(const _Self& __z) volatile {
    _M_re = __z._M_re;
    _M_im = __z._M_im;
    return *this;
  }

  __device__ 
  _Self& operator+= (const _Self& __z) {
    _M_re += __z._M_re;
    _M_im += __z._M_im;
    return *this;
  }

  __device__ 
  _Self& operator-= (const _Self& __z) {
    _M_re -= __z._M_re;
    _M_im -= __z._M_im;
    return *this;
  }

  __device__ 
  _Self& operator*= (const _Self& __z) {
    value_type __r = _M_re * __z._M_re - _M_im * __z._M_im;
    value_type __i = _M_re * __z._M_im + _M_im * __z._M_re;
    _M_re = __r;
    _M_im = __i;
    return *this;
  }

  __device__ 
  _Self& operator/= (const _Self& __z) {
    value_type __r;
    value_type __i;
    _div(_M_re, _M_im, __z._M_re, __z._M_im, __r, __i);
    _M_re = __r;
    _M_im = __i;
    return *this;
  }

  // Data members.
  value_type _M_re;
  value_type _M_im;
};

// Converting constructors from one of these three specialized types
// to another.

inline __device__ complex<float>::complex(const complex<double>& __z)
  : _M_re((float)__z._M_re), _M_im((float)__z._M_im) {}
inline __device__ complex<double>::complex(const complex<float>& __z)
  : _M_re(__z._M_re), _M_im(__z._M_im) {}

// Unary non-member arithmetic operators.

template <class _Tp>
inline complex<_Tp> __device__ operator+(const complex<_Tp>& __z)
{ return __z; }

template <class _Tp>
inline complex<_Tp> __device__  operator-(const complex<_Tp>& __z)
{ return complex<_Tp>(-__z._M_re, -__z._M_im); }

// Non-member arithmetic operations involving one real argument.

template <class _Tp>
__device__
inline complex<_Tp> operator+(const _Tp& __x, const complex<_Tp>& __z)
{ return complex<_Tp>(__x + __z._M_re, __z._M_im); }

template <class _Tp>
__device__
inline complex<_Tp> operator+(const complex<_Tp>& __z, const _Tp& __x)
{ return complex<_Tp>(__z._M_re + __x, __z._M_im); }

template <class _Tp>
__device__
inline complex<_Tp> operator-(const _Tp& __x, const complex<_Tp>& __z)
{ return complex<_Tp>(__x - __z._M_re, -__z._M_im); }

template <class _Tp>
__device__
inline complex<_Tp> operator-(const complex<_Tp>& __z, const _Tp& __x)
{ return complex<_Tp>(__z._M_re - __x, __z._M_im); }

template <class _Tp>
__device__
inline complex<_Tp> operator*(const _Tp& __x, const complex<_Tp>& __z)
{ return complex<_Tp>(__x * __z._M_re, __x * __z._M_im); }

template <class _Tp>
__device__
inline complex<_Tp> operator*(const complex<_Tp>& __z, const _Tp& __x)
{ return complex<_Tp>(__z._M_re * __x, __z._M_im * __x); }

template <class _Tp>
__device__
inline complex<_Tp> operator/(const _Tp& __x, const complex<_Tp>& __z) {
  complex<_Tp> __result;
  complex<_Tp>::_div(__x,
                     __z._M_re, __z._M_im,
                     __result._M_re, __result._M_im);
  return __result;
}

template <class _Tp>
__device__
inline complex<_Tp> operator/(const complex<_Tp>& __z, const _Tp& __x)
{ return complex<_Tp>(__z._M_re / __x, __z._M_im / __x); }

// Non-member arithmetic operations involving two complex arguments

template <class _Tp>
__device__
inline complex<_Tp>
operator+(const complex<_Tp>& __z1, const complex<_Tp>& __z2)
{ return complex<_Tp>(__z1._M_re + __z2._M_re, __z1._M_im + __z2._M_im); }

template <class _Tp>
__device__
inline complex<_Tp>
operator+(const volatile complex<_Tp>& __z1, const volatile complex<_Tp>& __z2)
{ return complex<_Tp>(__z1._M_re + __z2._M_re, __z1._M_im + __z2._M_im); }


template <class _Tp>
__device__
inline complex<_Tp> __device__
operator-(const complex<_Tp>& __z1, const complex<_Tp>& __z2)
{ return complex<_Tp>(__z1._M_re - __z2._M_re, __z1._M_im - __z2._M_im); }

template <class _Tp>
__device__
inline complex<_Tp> __device__
operator*(const complex<_Tp>& __z1, const complex<_Tp>& __z2) {
  return complex<_Tp>(__z1._M_re * __z2._M_re - __z1._M_im * __z2._M_im,
                      __z1._M_re * __z2._M_im + __z1._M_im * __z2._M_re);
}

template <class _Tp>
__device__
inline complex<_Tp> __device__
operator*(const volatile complex<_Tp>& __z1, const volatile complex<_Tp>& __z2) {
  return complex<_Tp>(__z1._M_re * __z2._M_re - __z1._M_im * __z2._M_im,
                      __z1._M_re * __z2._M_im + __z1._M_im * __z2._M_re);
}

template <class _Tp>
__device__
inline complex<_Tp> __device__
operator/(const complex<_Tp>& __z1, const complex<_Tp>& __z2) {
  complex<_Tp> __result;
  complex<_Tp>::_div(__z1._M_re, __z1._M_im,
                     __z2._M_re, __z2._M_im,
                     __result._M_re, __result._M_im);
  return __result;
}

// Comparison operators.

template <class _Tp>
__device__
inline bool operator==(const complex<_Tp>& __z1, const complex<_Tp>& __z2)
{ return __z1._M_re == __z2._M_re && __z1._M_im == __z2._M_im; }

template <class _Tp>
__device__
inline bool operator==(const complex<_Tp>& __z, const _Tp& __x)
{ return __z._M_re == __x && __z._M_im == 0; }

template <class _Tp>
__device__
inline bool operator==(const _Tp& __x, const complex<_Tp>& __z)
{ return __x == __z._M_re && 0 == __z._M_im; }

//04/27/04 dums: removal of this check, if it is restablish
//please explain why the other operators are not macro guarded
//#ifdef _STLP_FUNCTION_TMPL_PARTIAL_ORDER

template <class _Tp>
__device__
inline bool operator!=(const complex<_Tp>& __z1, const complex<_Tp>& __z2)
{ return __z1._M_re != __z2._M_re || __z1._M_im != __z2._M_im; }

//#endif /* _STLP_FUNCTION_TMPL_PARTIAL_ORDER */

template <class _Tp>
__device__
inline bool operator!=(const complex<_Tp>& __z, const _Tp& __x)
{ return __z._M_re != __x || __z._M_im != 0; }

template <class _Tp>
__device__
inline bool operator!=(const _Tp& __x, const complex<_Tp>& __z)
{ return __x != __z._M_re || 0 != __z._M_im; }

// Other basic arithmetic operations
template <class _Tp>
__device__
inline _Tp real(const complex<_Tp>& __z)
{ return __z._M_re; }

template <class _Tp>
__device__
inline _Tp imag(const complex<_Tp>& __z)
{ return __z._M_im; }

template <class _Tp>
__device__
_Tp abs(const complex<_Tp>& __z);

template <class _Tp>
__device__
_Tp arg(const complex<_Tp>& __z);

template <class _Tp>
__device__
inline _Tp norm(const complex<_Tp>& __z)
{ return __z._M_re * __z._M_re + __z._M_im * __z._M_im; }

template <class _Tp>
__device__
inline complex<_Tp> conj(const complex<_Tp>& __z)
{ return complex<_Tp>(__z._M_re, -__z._M_im); }

template <class _Tp>
__device__
complex<_Tp> polar(const _Tp& __rho)
{ return complex<_Tp>(__rho, 0); }

template <class _Tp>
__device__
complex<_Tp> polar(const _Tp& __rho, const _Tp& __phi);

_STLP_TEMPLATE_NULL
__device__ float abs(const complex<float>&);
_STLP_TEMPLATE_NULL
__device__ double abs(const complex<double>&);
_STLP_TEMPLATE_NULL
__device__ float arg(const complex<float>&);
_STLP_TEMPLATE_NULL
__device__ double arg(const complex<double>&);
_STLP_TEMPLATE_NULL
__device__ complex<float> polar(const float& __rho, const float& __phi);
_STLP_TEMPLATE_NULL
__device__ complex<double> polar(const double& __rho, const double& __phi);

template <class _Tp>
__device__
_Tp abs(const complex<_Tp>& __z)
{ return _Tp(abs(complex<double>(double(__z.real()), double(__z.imag())))); }

template <class _Tp>
__device__
_Tp arg(const complex<_Tp>& __z)
{ return _Tp(arg(complex<double>(double(__z.real()), double(__z.imag())))); }

template <class _Tp>
__device__
complex<_Tp> polar(const _Tp& __rho, const _Tp& __phi) {
  complex<double> __tmp = polar(double(__rho), double(__phi));
  return complex<_Tp>(_Tp(__tmp.real()), _Tp(__tmp.imag()));
}


// Transcendental functions.  These are defined only for float,
//  double, and long double.  (Sqrt isn't transcendental, of course,
//  but it's included in this section anyway.)

__device__ complex<float> sqrt(const complex<float>&);

__device__ complex<float> exp(const complex<float>&);
__device__ complex<float>  log(const complex<float>&);
__device__ complex<float> log10(const complex<float>&);

// uses some stlport-private power thing
// __device__ complex<float> pow(const complex<float>&, int);
__device__ complex<float> pow(const complex<float>&, const float&);
__device__ complex<float> pow(const float&, const complex<float>&);
__device__ complex<float> pow(const complex<float>&, const complex<float>&);

__device__ complex<float> sin(const complex<float>&);
__device__ complex<float> cos(const complex<float>&);
__device__ complex<float> tan(const complex<float>&);

__device__ complex<float> sinh(const complex<float>&);
__device__ complex<float> cosh(const complex<float>&);
__device__ complex<float> tanh(const complex<float>&);

__device__ complex<double> sqrt(const complex<double>&);

__device__ complex<double> exp(const complex<double>&);
__device__ complex<double> log(const complex<double>&);
__device__ complex<double> log10(const complex<double>&);

// uses some stlport-private power thing
// __device__ complex<double> pow(const complex<double>&, int);
__device__ complex<double> pow(const complex<double>&, const double&);
__device__ complex<double> pow(const double&, const complex<double>&);
__device__ complex<double> pow(const complex<double>&, const complex<double>&);

__device__ complex<double> sin(const complex<double>&);
__device__ complex<double> cos(const complex<double>&);
__device__ complex<double> tan(const complex<double>&);

__device__ complex<double> sinh(const complex<double>&);
__device__ complex<double> cosh(const complex<double>&);
__device__ complex<double> tanh(const complex<double>&);

}
}

#if 0 
#ifndef _STLP_LINK_TIME_INSTANTIATION
#  include <stl/_complex.c>
#endif
#endif

#include <pycuda-complex-impl.hpp>

#endif
