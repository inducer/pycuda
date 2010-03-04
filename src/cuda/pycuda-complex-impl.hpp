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
 */

#ifndef PYCUDA_COMPLEX_IMPL_HPP_SEEN
#define PYCUDA_COMPLEX_IMPL_HPP_SEEN

extern "C++" {
namespace pycuda {

// Complex division and square roots.

// Absolute value
_STLP_TEMPLATE_NULL
__device__ float abs(const complex<float>& __z)
{ return ::hypot(__z._M_re, __z._M_im); }
_STLP_TEMPLATE_NULL
__device__ double abs(const complex<double>& __z)
{ return ::hypot(__z._M_re, __z._M_im); }

// Phase

_STLP_TEMPLATE_NULL
__device__ float arg(const complex<float>& __z)
{ return ::atan2(__z._M_im, __z._M_re); }

_STLP_TEMPLATE_NULL
__device__ double arg(const complex<double>& __z)
{ return ::atan2(__z._M_im, __z._M_re); }

// Construct a complex number from polar representation
_STLP_TEMPLATE_NULL
__device__ complex<float> polar(const float& __rho, const float& __phi)
{ return complex<float>(__rho * ::cos(__phi), __rho * ::sin(__phi)); }
_STLP_TEMPLATE_NULL
__device__ complex<double> polar(const double& __rho, const double& __phi)
{ return complex<double>(__rho * ::cos(__phi), __rho * ::sin(__phi)); }

// Division
template <class _Tp>
__device__
static void _divT(const _Tp& __z1_r, const _Tp& __z1_i,
                  const _Tp& __z2_r, const _Tp& __z2_i,
                  _Tp& __res_r, _Tp& __res_i) {
  _Tp __ar = __z2_r >= 0 ? __z2_r : -__z2_r;
  _Tp __ai = __z2_i >= 0 ? __z2_i : -__z2_i;

  if (__ar <= __ai) {
    _Tp __ratio = __z2_r / __z2_i;
    _Tp __denom = __z2_i * (1 + __ratio * __ratio);
    __res_r = (__z1_r * __ratio + __z1_i) / __denom;
    __res_i = (__z1_i * __ratio - __z1_r) / __denom;
  }
  else {
    _Tp __ratio = __z2_i / __z2_r;
    _Tp __denom = __z2_r * (1 + __ratio * __ratio);
    __res_r = (__z1_r + __z1_i * __ratio) / __denom;
    __res_i = (__z1_i - __z1_r * __ratio) / __denom;
  }
}

template <class _Tp>
__device__
static void _divT(const _Tp& __z1_r,
                  const _Tp& __z2_r, const _Tp& __z2_i,
                  _Tp& __res_r, _Tp& __res_i) {
  _Tp __ar = __z2_r >= 0 ? __z2_r : -__z2_r;
  _Tp __ai = __z2_i >= 0 ? __z2_i : -__z2_i;

  if (__ar <= __ai) {
    _Tp __ratio = __z2_r / __z2_i;
    _Tp __denom = __z2_i * (1 + __ratio * __ratio);
    __res_r = (__z1_r * __ratio) / __denom;
    __res_i = - __z1_r / __denom;
  }
  else {
    _Tp __ratio = __z2_i / __z2_r;
    _Tp __denom = __z2_r * (1 + __ratio * __ratio);
    __res_r = __z1_r / __denom;
    __res_i = - (__z1_r * __ratio) / __denom;
  }
}

__device__
void
complex<float>::_div(const float& __z1_r, const float& __z1_i,
                     const float& __z2_r, const float& __z2_i,
                     float& __res_r, float& __res_i)
{ _divT(__z1_r, __z1_i, __z2_r, __z2_i, __res_r, __res_i); }

__device__
void
complex<float>::_div(const float& __z1_r,
                     const float& __z2_r, const float& __z2_i,
                     float& __res_r, float& __res_i)
{ _divT(__z1_r, __z2_r, __z2_i, __res_r, __res_i); }


__device__
void
complex<double>::_div(const double& __z1_r, const double& __z1_i,
                      const double& __z2_r, const double& __z2_i,
                      double& __res_r, double& __res_i)
{ _divT(__z1_r, __z1_i, __z2_r, __z2_i, __res_r, __res_i); }

__device__
void
complex<double>::_div(const double& __z1_r,
                      const double& __z2_r, const double& __z2_i,
                      double& __res_r, double& __res_i)
{ _divT(__z1_r, __z2_r, __z2_i, __res_r, __res_i); }

//----------------------------------------------------------------------
// Square root
template <class _Tp>
__device__ 
static complex<_Tp> sqrtT(const complex<_Tp>& z) {
  _Tp re = z._M_re;
  _Tp im = z._M_im;
  _Tp mag = ::hypot(re, im);
  complex<_Tp> result;

  if (mag == 0.f) {
    result._M_re = result._M_im = 0.f;
  } else if (re > 0.f) {
    result._M_re = ::sqrt(0.5f * (mag + re));
    result._M_im = im/result._M_re/2.f;
  } else {
    result._M_im = ::sqrt(0.5f * (mag - re));
    if (im < 0.f)
      result._M_im = - result._M_im;
    result._M_re = im/result._M_im/2.f;
  }
  return result;
}

__device__
complex<float> 
sqrt(const complex<float>& z) { return sqrtT(z); }

__device__
complex<double>
sqrt(const complex<double>& z) { return sqrtT(z); }

// exp, log, pow for complex<float>, complex<double>, and complex<long double>
//----------------------------------------------------------------------
// exp
template <class _Tp>
__device__ 
static complex<_Tp> expT(const complex<_Tp>& z) {
  _Tp expx = ::exp(z._M_re);
  return complex<_Tp>(expx * ::cos(z._M_im),
                      expx * ::sin(z._M_im));
}
__device__ complex<float>  exp(const complex<float>& z)
{ return expT(z); }

__device__ complex<double> exp(const complex<double>& z)
{ return expT(z); }

#if 0
//----------------------------------------------------------------------
// log10
template <class _Tp>
static __device__ complex<_Tp> log10T(const complex<_Tp>& z, const _Tp& ln10_inv) {
  complex<_Tp> r;

  r._M_im = ::atan2(z._M_im, z._M_re) * ln10_inv;
  r._M_re = ::log10(::hypot(z._M_re, z._M_im));
  return r;
}

static const float LN10_INVF = 1.f / ::log(10.f);
__device__ complex<float> log10(const complex<float>& z)
{ return log10T(z, LN10_INVF); }

static const double LN10_INV = 1. / ::log10(10.);
__device__ complex<double> log10(const complex<double>& z)
{ return log10T(z, LN10_INV); }

#endif

//----------------------------------------------------------------------
// log
template <class _Tp>
static __device__ complex<_Tp> logT(const complex<_Tp>& z) {
  complex<_Tp> r;

  r._M_im = ::atan2(z._M_im, z._M_re);
  r._M_re = ::log(::hypot(z._M_re, z._M_im));
  return r;
}
__device__ complex<float> log(const complex<float>& z)
{ return logT(z); }

__device__ complex<double> log(const complex<double>& z)
{ return logT(z); }

#if 0
//----------------------------------------------------------------------
// pow
template <class _Tp>
__device__
static complex<_Tp> powT(const _Tp& a, const complex<_Tp>& b) {
  _Tp logr = ::log(a);
  _Tp x = ::exp(logr * b._M_re);
  _Tp y = logr * b._M_im;

  return complex<_Tp>(x * ::cos(y), x * ::sin(y));
}

template <class _Tp>
__device__
static complex<_Tp> powT(const complex<_Tp>& z_in, int n) {
  complex<_Tp> z = z_in;
  z = _STLP_PRIV __power(z, (n < 0 ? -n : n), multiplies< complex<_Tp> >());
  if (n < 0)
    return _Tp(1.0) / z;
  else
    return z;
}

template <class _Tp>
__device__
static complex<_Tp> powT(const complex<_Tp>& a, const _Tp& b) {
  _Tp logr = ::log(::hypot(a._M_re,a._M_im));
  _Tp logi = ::atan2(a._M_im, a._M_re);
  _Tp x = ::exp(logr * b);
  _Tp y = logi * b;

  return complex<_Tp>(x * ::cos(y), x * ::sin(y));
}

template <class _Tp>
__device__
static complex<_Tp> powT(const complex<_Tp>& a, const complex<_Tp>& b) {
  _Tp logr = ::log(::hypot(a._M_re,a._M_im));
  _Tp logi = ::atan2(a._M_im, a._M_re);
  _Tp x = ::exp(logr * b._M_re - logi * b._M_im);
  _Tp y = logr * b._M_im + logi * b._M_re;

  return complex<_Tp>(x * ::cos(y), x * ::sin(y));
}

__device__ complex<float> pow(const float& a, const complex<float>& b)
{ return powT(a, b); }

__device__ complex<float> pow(const complex<float>& z_in, int n)
{ return powT(z_in, n); }

__device__ complex<float> pow(const complex<float>& a, const float& b)
{ return powT(a, b); }

__device__ complex<float> pow(const complex<float>& a, const complex<float>& b)
{ return powT(a, b); }

__device__ complex<double> pow(const double& a, const complex<double>& b)
{ return powT(a, b); }

__device__ complex<double> pow(const complex<double>& z_in, int n)
{ return powT(z_in, n); }

__device__ complex<double> pow(const complex<double>& a, const double& b)
{ return powT(a, b); }

__device__ complex<double> pow(const complex<double>& a, const complex<double>& b)
{ return powT(a, b); }

#endif

}
}

#endif
