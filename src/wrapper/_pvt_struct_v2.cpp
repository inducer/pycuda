/* struct module -- pack values into and (out of) strings */

/* New version supporting byte order, alignment and size options,
   character strings, and unsigned numbers */

/* Compared with vanilla Python's struct module, this adds support
 * for packing complex values and only supports native packing.
 * (the minimum that's needed for PyOpenCL/PyCUDA.) */

#define PY_SSIZE_T_CLEAN

#include "Python.h"
#include "structseq.h"
#include "structmember.h"
#include <ctype.h>
#include "numpy_init.hpp"

// static PyTypeObject PyStructType;

/* compatibility macros */
#if (PY_VERSION_HEX < 0x02050000)
#ifndef PY_SSIZE_T_MIN
typedef long int Py_ssize_t;
#endif

#define PyInt_FromSsize_t(x) PyInt_FromLong(x)
#define PyInt_AsSsize_t(x) PyInt_AsLong(x)
#endif

/* If PY_STRUCT_FLOAT_COERCE is defined, the struct module will allow float
   arguments for integer formats with a warning for backwards
   compatibility. */

#define PY_STRUCT_FLOAT_COERCE 1

#ifdef PY_STRUCT_FLOAT_COERCE
#define FLOAT_COERCE "integer argument expected, got float"
#endif

/* Compatibility with Py2.5 and older */

#ifndef Py_TYPE
#  define Py_TYPE(o) ((o)->ob_type)
#endif

#ifndef PyVarObject_HEAD_INIT
#define PyVarObject_HEAD_INIT(type, size)       \
          PyObject_HEAD_INIT(type) size,
#endif

#ifndef SIZEOF_SIZE_T
#define SIZEOF_SIZE_T sizeof(size_t)
#endif

#ifndef PY_SSIZE_T_MAX
#define PY_SSIZE_T_MAX LONG_MAX
#endif

/* The translation function for each format character is table driven */
typedef struct _formatdef {
	char format;
	Py_ssize_t size;
	Py_ssize_t alignment;
	PyObject* (*unpack)(const char *,
			    const struct _formatdef *);
	int (*pack)(char *, PyObject *,
		    const struct _formatdef *);
} formatdef;

typedef struct _formatcode {
	const struct _formatdef *fmtdef;
	Py_ssize_t offset;
	Py_ssize_t size;
} formatcode;

/* Struct object interface */

typedef struct {
	PyObject_HEAD
	Py_ssize_t s_size;
	Py_ssize_t s_len;
	formatcode *s_codes;
	PyObject *s_format;
	PyObject *weakreflist; /* List of weak references */
} PyStructObject;


#define PyStruct_Check(op) PyObject_TypeCheck(op, &PyStructType)
#define PyStruct_CheckExact(op) (Py_TYPE(op) == &PyStructType)


/* Exception */

static PyObject *StructError;


/* Define various structs to figure out the alignments of types */


typedef struct { char c; short x; } st_short;
typedef struct { char c; int x; } st_int;
typedef struct { char c; long x; } st_long;
typedef struct { char c; float x; } st_float;
typedef struct { char c; double x; } st_double;
typedef struct { char c; void *x; } st_void_p;

#define SHORT_ALIGN (sizeof(st_short) - sizeof(short))
#define INT_ALIGN (sizeof(st_int) - sizeof(int))
#define LONG_ALIGN (sizeof(st_long) - sizeof(long))
#define FLOAT_ALIGN (sizeof(st_float) - sizeof(float))
#define DOUBLE_ALIGN (sizeof(st_double) - sizeof(double))
#define VOID_P_ALIGN (sizeof(st_void_p) - sizeof(void *))

/* We can't support q and Q in native mode unless the compiler does;
   in std mode, they're 8 bytes on all platforms. */
#ifdef HAVE_LONG_LONG
typedef struct { char c; PY_LONG_LONG x; } s_long_long;
#define LONG_LONG_ALIGN (sizeof(s_long_long) - sizeof(PY_LONG_LONG))
#endif

#define BOOL_TYPE bool
typedef struct { char c; bool x; } s_bool;
#define BOOL_ALIGN (sizeof(s_bool) - sizeof(BOOL_TYPE))

#define STRINGIFY(x)    #x

#ifdef __powerc
#pragma options align=reset
#endif

static char *integer_codes = "bBhHiIlLqQ";

static void s_dealloc(PyStructObject *s);
static int s_init(PyObject *self, PyObject *args, PyObject *kwds);
static PyObject *s_new(PyTypeObject *type, PyObject *args, PyObject *kwds);
static PyObject *s_pack(PyObject *self, PyObject *args);
static PyObject *s_pack_into(PyObject *self, PyObject *args);
static PyObject *s_unpack(PyObject *self, PyObject *inputstr);
static PyObject *s_unpack_from(PyObject *self, PyObject *args, PyObject *kwds);
static PyObject *s_get_format(PyStructObject *self, void *unused);
static PyObject *s_get_size(PyStructObject *self, void *unused);

PyDoc_STRVAR(s__doc__, "Compiled struct object");

/* List of functions */

PyDoc_STRVAR(s_pack__doc__,
"S.pack(v1, v2, ...) -> string\n\
\n\
Return a string containing values v1, v2, ... packed according to this\n\
Struct's format. See struct.__doc__ for more on format strings.");

PyDoc_STRVAR(s_pack_into__doc__,
"S.pack_into(buffer, offset, v1, v2, ...)\n\
\n\
Pack the values v1, v2, ... according to this Struct's format, write \n\
the packed bytes into the writable buffer buf starting at offset.  Note\n\
that the offset is not an optional argument.  See struct.__doc__ for \n\
more on format strings.");

PyDoc_STRVAR(s_unpack__doc__,
"S.unpack(str) -> (v1, v2, ...)\n\
\n\
Return tuple containing values unpacked according to this Struct's format.\n\
Requires len(str) == self.size. See struct.__doc__ for more on format\n\
strings.");

PyDoc_STRVAR(s_unpack_from__doc__,
"S.unpack_from(buffer[, offset]) -> (v1, v2, ...)\n\
\n\
Return tuple containing values unpacked according to this Struct's format.\n\
Unlike unpack, unpack_from can unpack values from any object supporting\n\
the buffer API, not just str. Requires len(buffer[offset:]) >= self.size.\n\
See struct.__doc__ for more on format strings.");


static struct PyMethodDef s_methods[] = {
	{"pack",	s_pack,		METH_VARARGS, s_pack__doc__},
	{"pack_into",	s_pack_into,	METH_VARARGS, s_pack_into__doc__},
	{"unpack",	s_unpack,       METH_O, s_unpack__doc__},
	{"unpack_from",	(PyCFunction)s_unpack_from, METH_VARARGS|METH_KEYWORDS,
			s_unpack_from__doc__},
	{NULL,	 NULL}		/* sentinel */
};

#define OFF(x) offsetof(PyStructObject, x)

static PyGetSetDef s_getsetlist[] = {
	{"format", (getter)s_get_format, (setter)NULL, "struct format string", NULL},
	{"size", (getter)s_get_size, (setter)NULL, "struct size in bytes", NULL},
	{NULL} /* sentinel */
};

static
PyTypeObject PyStructType = {
	PyVarObject_HEAD_INIT(NULL, 0)
	"Struct",
	sizeof(PyStructObject),
	0,
	(destructor)s_dealloc,	/* tp_dealloc */
	0,					/* tp_print */
	0,					/* tp_getattr */
	0,					/* tp_setattr */
	0,					/* tp_compare */
	0,					/* tp_repr */
	0,					/* tp_as_number */
	0,					/* tp_as_sequence */
	0,					/* tp_as_mapping */
	0,					/* tp_hash */
	0,					/* tp_call */
	0,					/* tp_str */
	PyObject_GenericGetAttr,	/* tp_getattro */
	PyObject_GenericSetAttr,	/* tp_setattro */
	0,					/* tp_as_buffer */
	Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE | Py_TPFLAGS_HAVE_WEAKREFS,/* tp_flags */
	s__doc__,			/* tp_doc */
	0,					/* tp_traverse */
	0,					/* tp_clear */
	0,					/* tp_richcompare */
	offsetof(PyStructObject, weakreflist),	/* tp_weaklistoffset */
	0,					/* tp_iter */
	0,					/* tp_iternext */
	s_methods,			/* tp_methods */
	NULL,				/* tp_members */
	s_getsetlist,		/* tp_getset */
	0,					/* tp_base */
	0,					/* tp_dict */
	0,					/* tp_descr_get */
	0,					/* tp_descr_set */
	0,					/* tp_dictoffset */
	s_init,				/* tp_init */
	PyType_GenericAlloc,/* tp_alloc */
	s_new,				/* tp_new */
	PyObject_Del,		/* tp_free */
};

/* Helper to get a PyLongObject by hook or by crook.  Caller should decref. */

static PyObject *
get_pylong(PyObject *v)
{
        PyNumberMethods *m;

	assert(v != NULL);
	if (PyInt_Check(v))
		return PyLong_FromLong(PyInt_AS_LONG(v));
	if (PyLong_Check(v)) {
		Py_INCREF(v);
		return v;
	}

        m = Py_TYPE(v)->tp_as_number;
        if (m != NULL && m->nb_long != NULL) {
                v = m->nb_long(v);
                if (v == NULL)
                        return NULL;
                if (PyLong_Check(v))
                        return v;
                Py_DECREF(v);
        }

	PyErr_SetString(StructError,
			"cannot convert argument to long");
	return NULL;
}

/* Helper to convert a Python object to a C long.  Sets an exception
   (struct.error for an inconvertible type, OverflowError for
   out-of-range values) and returns -1 on error. */

static int
get_long(PyObject *v, long *p)
{
	long x;

	v = get_pylong(v);
	if (v == NULL)
		return -1;
	assert(PyLong_Check(v));
	x = PyLong_AsLong(v);
	Py_DECREF(v);
	if (x == (long)-1 && PyErr_Occurred())
		return -1;
	*p = x;
	return 0;
}

/* Same, but handling unsigned long */

static int
get_ulong(PyObject *v, unsigned long *p)
{
	unsigned long x;

	v = get_pylong(v);
	if (v == NULL)
		return -1;
	assert(PyLong_Check(v));
	x = PyLong_AsUnsignedLong(v);
	Py_DECREF(v);
	if (x == (unsigned long)-1 && PyErr_Occurred())
		return -1;
	*p = x;
	return 0;
}

#ifdef HAVE_LONG_LONG

/* Same, but handling native long long. */

static int
get_longlong(PyObject *v, PY_LONG_LONG *p)
{
	PY_LONG_LONG x;

	v = get_pylong(v);
	if (v == NULL)
		return -1;
	assert(PyLong_Check(v));
	x = PyLong_AsLongLong(v);
	Py_DECREF(v);
	if (x == (PY_LONG_LONG)-1 && PyErr_Occurred())
		return -1;
	*p = x;
	return 0;
}

/* Same, but handling native unsigned long long. */

static int
get_ulonglong(PyObject *v, unsigned PY_LONG_LONG *p)
{
	unsigned PY_LONG_LONG x;

	v = get_pylong(v);
	if (v == NULL)
		return -1;
	assert(PyLong_Check(v));
	x = PyLong_AsUnsignedLongLong(v);
	Py_DECREF(v);
	if (x == (unsigned PY_LONG_LONG)-1 && PyErr_Occurred())
		return -1;
	*p = x;
	return 0;
}

#endif

#if (SIZEOF_LONG > SIZEOF_INT)

/* Helper to format the range error exceptions */
static int
_range_error(const formatdef *f, int is_unsigned)
{
	/* ulargest is the largest unsigned value with f->size bytes.
	 * Note that the simpler:
	 *     ((size_t)1 << (f->size * 8)) - 1
	 * doesn't work when f->size == sizeof(size_t) because C doesn't
	 * define what happens when a left shift count is >= the number of
	 * bits in the integer being shifted; e.g., on some boxes it doesn't
	 * shift at all when they're equal.
	 */
	const size_t ulargest = (size_t)-1 >> ((SIZEOF_SIZE_T - f->size)*8);
	assert(f->size >= 1 && f->size <= SIZEOF_SIZE_T);
	if (is_unsigned)
		PyErr_Format(StructError,
			"'%c' format requires 0 <= number <= %zu",
			f->format,
			ulargest);
	else {
		const Py_ssize_t largest = (Py_ssize_t)(ulargest >> 1);
		PyErr_Format(StructError,
			"'%c' format requires %zd <= number <= %zd",
			f->format,
			~ largest,
			largest);
	}
	return -1;
}

#endif


/* A large number of small routines follow, with names of the form

   [bln][up]_TYPE

   [bln] distiguishes among big-endian, little-endian and native.
   [pu] distiguishes between pack (to struct) and unpack (from struct).
   TYPE is one of char, byte, ubyte, etc.
*/

/* Native mode routines. ****************************************************/
/* NOTE:
   In all n[up]_<type> routines handling types larger than 1 byte, there is
   *no* guarantee that the p pointer is properly aligned for each type,
   therefore memcpy is called.  An intermediate variable is used to
   compensate for big-endian architectures.
   Normally both the intermediate variable and the memcpy call will be
   skipped by C optimisation in little-endian architectures (gcc >= 2.91
   does this). */

static PyObject *
nu_char(const char *p, const formatdef *f)
{
	return PyString_FromStringAndSize(p, 1);
}

static PyObject *
nu_byte(const char *p, const formatdef *f)
{
	return PyInt_FromLong((long) *(signed char *)p);
}

static PyObject *
nu_ubyte(const char *p, const formatdef *f)
{
	return PyInt_FromLong((long) *(unsigned char *)p);
}

static PyObject *
nu_short(const char *p, const formatdef *f)
{
	short x;
	memcpy((char *)&x, p, sizeof x);
	return PyInt_FromLong((long)x);
}

static PyObject *
nu_ushort(const char *p, const formatdef *f)
{
	unsigned short x;
	memcpy((char *)&x, p, sizeof x);
	return PyInt_FromLong((long)x);
}

static PyObject *
nu_int(const char *p, const formatdef *f)
{
	int x;
	memcpy((char *)&x, p, sizeof x);
	return PyInt_FromLong((long)x);
}

static PyObject *
nu_uint(const char *p, const formatdef *f)
{
	unsigned int x;
	memcpy((char *)&x, p, sizeof x);
#if (SIZEOF_LONG > SIZEOF_INT)
	return PyInt_FromLong((long)x);
#else
	if (x <= ((unsigned int)LONG_MAX))
		return PyInt_FromLong((long)x);
	return PyLong_FromUnsignedLong((unsigned long)x);
#endif
}

static PyObject *
nu_long(const char *p, const formatdef *f)
{
	long x;
	memcpy((char *)&x, p, sizeof x);
	return PyInt_FromLong(x);
}

static PyObject *
nu_ulong(const char *p, const formatdef *f)
{
	unsigned long x;
	memcpy((char *)&x, p, sizeof x);
	if (x <= LONG_MAX)
		return PyInt_FromLong((long)x);
	return PyLong_FromUnsignedLong(x);
}

/* Native mode doesn't support q or Q unless the platform C supports
   long long (or, on Windows, __int64). */

#ifdef HAVE_LONG_LONG

static PyObject *
nu_longlong(const char *p, const formatdef *f)
{
	PY_LONG_LONG x;
	memcpy((char *)&x, p, sizeof x);
	if (x >= LONG_MIN && x <= LONG_MAX)
		return PyInt_FromLong(Py_SAFE_DOWNCAST(x, PY_LONG_LONG, long));
	return PyLong_FromLongLong(x);
}

static PyObject *
nu_ulonglong(const char *p, const formatdef *f)
{
	unsigned PY_LONG_LONG x;
	memcpy((char *)&x, p, sizeof x);
	if (x <= LONG_MAX)
		return PyInt_FromLong(Py_SAFE_DOWNCAST(x, unsigned PY_LONG_LONG, long));
	return PyLong_FromUnsignedLongLong(x);
}

#endif

static PyObject *
nu_bool(const char *p, const formatdef *f)
{
	BOOL_TYPE x;
	memcpy((char *)&x, p, sizeof x);
	return PyBool_FromLong(x != 0);
}


static PyObject *
nu_float(const char *p, const formatdef *f)
{
	float x;
	memcpy((char *)&x, p, sizeof x);
	return PyFloat_FromDouble((double)x);
}

static PyObject *
nu_double(const char *p, const formatdef *f)
{
	double x;
	memcpy((char *)&x, p, sizeof x);
	return PyFloat_FromDouble(x);
}

static PyObject *
nu_complex_float(const char *p, const formatdef *f)
{
	float re, im;
	memcpy((char *)&re, p, sizeof re);
	memcpy((char *)&im, p+sizeof re, sizeof im);
	return PyComplex_FromDoubles((double)re, (double) im);
}

static PyObject *
nu_complex_double(const char *p, const formatdef *f)
{
	double re, im;
	memcpy((char *)&re, p, sizeof re);
	memcpy((char *)&im, p+sizeof re, sizeof im);
	return PyComplex_FromDoubles(re, im);
}

static PyObject *
nu_void_p(const char *p, const formatdef *f)
{
	void *x;
	memcpy((char *)&x, p, sizeof x);
	return PyLong_FromVoidPtr(x);
}

static int
np_byte(char *p, PyObject *v, const formatdef *f)
{
	long x;
	if (get_long(v, &x) < 0)
		return -1;
	if (x < -128 || x > 127){
		PyErr_SetString(StructError,
				"byte format requires -128 <= number <= 127");
		return -1;
	}
	*p = (char)x;
	return 0;
}

static int
np_ubyte(char *p, PyObject *v, const formatdef *f)
{
	long x;
	if (get_long(v, &x) < 0)
		return -1;
	if (x < 0 || x > 255){
		PyErr_SetString(StructError,
				"ubyte format requires 0 <= number <= 255");
		return -1;
	}
	*p = (char)x;
	return 0;
}

static int
np_char(char *p, PyObject *v, const formatdef *f)
{
	if (!PyString_Check(v) || PyString_Size(v) != 1) {
		PyErr_SetString(StructError,
				"char format require string of length 1");
		return -1;
	}
	*p = *PyString_AsString(v);
	return 0;
}

static int
np_short(char *p, PyObject *v, const formatdef *f)
{
	long x;
	short y;
	if (get_long(v, &x) < 0)
		return -1;
	if (x < SHRT_MIN || x > SHRT_MAX){
		PyErr_SetString(StructError,
				"short format requires " STRINGIFY(SHRT_MIN)
				" <= number <= " STRINGIFY(SHRT_MAX));
		return -1;
	}
	y = (short)x;
	memcpy(p, (char *)&y, sizeof y);
	return 0;
}

static int
np_ushort(char *p, PyObject *v, const formatdef *f)
{
	long x;
	unsigned short y;
	if (get_long(v, &x) < 0)
		return -1;
	if (x < 0 || x > USHRT_MAX){
		PyErr_SetString(StructError,
				"ushort format requires 0 <= number <= " STRINGIFY(USHRT_MAX));
		return -1;
	}
	y = (unsigned short)x;
	memcpy(p, (char *)&y, sizeof y);
	return 0;
}

static int
np_int(char *p, PyObject *v, const formatdef *f)
{
	long x;
	int y;
	if (get_long(v, &x) < 0)
		return -1;
#if (SIZEOF_LONG > SIZEOF_INT)
	if ((x < ((long)INT_MIN)) || (x > ((long)INT_MAX)))
		return _range_error(f, 0);
#endif
	y = (int)x;
	memcpy(p, (char *)&y, sizeof y);
	return 0;
}

static int
np_uint(char *p, PyObject *v, const formatdef *f)
{
	unsigned long x;
	unsigned int y;
	if (get_ulong(v, &x) < 0)
		return -1;
	y = (unsigned int)x;
#if (SIZEOF_LONG > SIZEOF_INT)
	if (x > ((unsigned long)UINT_MAX))
		return _range_error(f, 1);
#endif
	memcpy(p, (char *)&y, sizeof y);
	return 0;
}

static int
np_long(char *p, PyObject *v, const formatdef *f)
{
	long x;
	if (get_long(v, &x) < 0)
		return -1;
	memcpy(p, (char *)&x, sizeof x);
	return 0;
}

static int
np_ulong(char *p, PyObject *v, const formatdef *f)
{
	unsigned long x;
	if (get_ulong(v, &x) < 0)
		return -1;
	memcpy(p, (char *)&x, sizeof x);
	return 0;
}

#ifdef HAVE_LONG_LONG

static int
np_longlong(char *p, PyObject *v, const formatdef *f)
{
	PY_LONG_LONG x;
	if (get_longlong(v, &x) < 0)
		return -1;
	memcpy(p, (char *)&x, sizeof x);
	return 0;
}

static int
np_ulonglong(char *p, PyObject *v, const formatdef *f)
{
	unsigned PY_LONG_LONG x;
	if (get_ulonglong(v, &x) < 0)
		return -1;
	memcpy(p, (char *)&x, sizeof x);
	return 0;
}
#endif


static int
np_bool(char *p, PyObject *v, const formatdef *f)
{
	BOOL_TYPE y; 
	y = PyObject_IsTrue(v) != 0;
	memcpy(p, (char *)&y, sizeof y);
	return 0;
}

static int
np_float(char *p, PyObject *v, const formatdef *f)
{
	float x = (float)PyFloat_AsDouble(v);
	if (x == -1 && PyErr_Occurred()) {
		PyErr_SetString(StructError,
				"required argument is not a float");
		return -1;
	}
	memcpy(p, (char *)&x, sizeof x);
	return 0;
}

static int
np_double(char *p, PyObject *v, const formatdef *f)
{
	double x = PyFloat_AsDouble(v);
	if (x == -1 && PyErr_Occurred()) {
		PyErr_SetString(StructError,
				"required argument is not a float");
		return -1;
	}
	memcpy(p, (char *)&x, sizeof(double));
	return 0;
}

static int
np_complex_float(char *p, PyObject *v, const formatdef *f)
{
        if (PyArray_IsZeroDim(v)) {
		PyObject *v_cast = PyArray_Cast(
				reinterpret_cast<PyArrayObject *>(v),
				NPY_CFLOAT);
		if (!v_cast)
			return -1;
		memcpy(p, PyArray_DATA(v_cast), PyArray_NBYTES(v_cast));
		Py_DECREF(v_cast);
	}
	else {
		float re = 0.0f;
		float im = 0.0f;
		Py_complex cplx;
#if (PY_VERSION_HEX < 0x02060000)
			if (PyComplex_Check(v))
				cplx = PyComplex_AsCComplex(v);
			else if (PyObject_HasAttrString(v, "__complex__"))
			{
				PyObject *v2 = PyObject_CallMethod(v, "__complex__", "");
				cplx = PyComplex_AsCComplex(v2);
				Py_DECREF(v2);
			}
			else
				cplx = PyComplex_AsCComplex(v);
#else
			cplx = PyComplex_AsCComplex(v);
#endif
		if (PyErr_Occurred()) {
			PyErr_SetString(StructError,
					"required argument is not a complex");
			return -1;
		}

		re = (float)cplx.real;
		im = (float)cplx.imag;
		memcpy(p, (char *)&re, sizeof re);
		memcpy(p+sizeof re, (char *)&im, sizeof im);
	}
	return 0;
}

static int
np_complex_double(char *p, PyObject *v, const formatdef *f)
{
        if (PyArray_IsZeroDim(v)) {
		PyObject *v_cast = PyArray_Cast(
				reinterpret_cast<PyArrayObject *>(v),
				NPY_CDOUBLE);
		if (!v_cast)
			return -1;
		memcpy(p, PyArray_DATA(v_cast), PyArray_NBYTES(v_cast));
		Py_DECREF(v_cast);
	}
	else {
		double re = 0.0;
		double im = 0.0;
		Py_complex cplx;
#if (PY_VERSION_HEX < 0x02060000)
			if (PyComplex_Check(v))
				cplx = PyComplex_AsCComplex(v);
			else if (PyObject_HasAttrString(v, "__complex__"))
			{
				PyObject *v2 = PyObject_CallMethod(v, "__complex__", "");
				cplx = PyComplex_AsCComplex(v2);
				Py_DECREF(v2);
			}
			else
				cplx = PyComplex_AsCComplex(v);
#else
			cplx = PyComplex_AsCComplex(v);
#endif
		if (PyErr_Occurred()) {
			PyErr_SetString(StructError,
					"required argument is not a complex");
			return -1;
		}
		re = cplx.real;
		im = cplx.imag;
		memcpy(p, (char *)&re, sizeof re);
		memcpy(p+sizeof re, (char *)&im, sizeof im);
	}
	return 0;
}

static int
np_void_p(char *p, PyObject *v, const formatdef *f)
{
	void *x;

	v = get_pylong(v);
	if (v == NULL)
		return -1;
	assert(PyLong_Check(v));
	x = PyLong_AsVoidPtr(v);
	Py_DECREF(v);
	if (x == NULL && PyErr_Occurred())
		return -1;
	memcpy(p, (char *)&x, sizeof x);
	return 0;
}

static formatdef native_table[] = {
	{'x',	sizeof(char),	0,		NULL},
	{'b',	sizeof(char),	0,		nu_byte,	np_byte},
	{'B',	sizeof(char),	0,		nu_ubyte,	np_ubyte},
	{'c',	sizeof(char),	0,		nu_char,	np_char},
	{'s',	sizeof(char),	0,		NULL},
	{'p',	sizeof(char),	0,		NULL},
	{'h',	sizeof(short),	SHORT_ALIGN,	nu_short,	np_short},
	{'H',	sizeof(short),	SHORT_ALIGN,	nu_ushort,	np_ushort},
	{'i',	sizeof(int),	INT_ALIGN,	nu_int,		np_int},
	{'I',	sizeof(int),	INT_ALIGN,	nu_uint,	np_uint},
	{'l',	sizeof(long),	LONG_ALIGN,	nu_long,	np_long},
	{'L',	sizeof(long),	LONG_ALIGN,	nu_ulong,	np_ulong},
#ifdef HAVE_LONG_LONG
	{'q',	sizeof(PY_LONG_LONG), LONG_LONG_ALIGN, nu_longlong, np_longlong},
	{'Q',	sizeof(PY_LONG_LONG), LONG_LONG_ALIGN, nu_ulonglong,np_ulonglong},
#endif
	{'?',	sizeof(BOOL_TYPE),	BOOL_ALIGN,	nu_bool,	np_bool},
	{'f',	sizeof(float),	FLOAT_ALIGN,	nu_float,	np_float},
	{'d',	sizeof(double),	DOUBLE_ALIGN,	nu_double,	np_double},
	{'F',	2*sizeof(float),	FLOAT_ALIGN,	nu_complex_float,	np_complex_float},
	{'D',	2*sizeof(double),	DOUBLE_ALIGN,	nu_complex_double,	np_complex_double},
	{'P',	sizeof(void *),	VOID_P_ALIGN,	nu_void_p,	np_void_p},
	{0}
};

/* Get the table entry for a format code */

static const formatdef *
getentry(int c, const formatdef *f)
{
	for (; f->format != '\0'; f++) {
		if (f->format == c) {
			return f;
		}
	}
	PyErr_SetString(StructError, "bad char in struct format");
	return NULL;
}


/* Align a size according to a format code */

static Py_ssize_t
align(Py_ssize_t size, char c, const formatdef *e)
{
	if (e->format == c) {
		if (e->alignment) {
			size = ((size + e->alignment - 1)
				/ e->alignment)
				* e->alignment;
		}
	}
	return size;
}


/* calculate the size of a format string */

static int
prepare_s(PyStructObject *self)
{
	const formatdef *f;
	const formatdef *e;
	formatcode *codes;

	const char *s;
	const char *fmt;
	char c;
	Py_ssize_t size, len, num, itemsize, x;

	fmt = PyString_AS_STRING(self->s_format);

	f = native_table;

	s = fmt;
	size = 0;
	len = 0;
	while ((c = *s++) != '\0') {
		if (isspace(Py_CHARMASK(c)))
			continue;
		if ('0' <= c && c <= '9') {
			num = c - '0';
			while ('0' <= (c = *s++) && c <= '9') {
				x = num*10 + (c - '0');
				if (x/10 != num) {
					PyErr_SetString(
						StructError,
						"overflow in item count");
					return -1;
				}
				num = x;
			}
			if (c == '\0')
				break;
		}
		else
			num = 1;

		e = getentry(c, f);
		if (e == NULL)
			return -1;

		switch (c) {
			case 's': /* fall through */
			case 'p': len++; break;
			case 'x': break;
			default: len += num; break;
		}

		itemsize = e->size;
		size = align(size, c, e);
		x = num * itemsize;
		size += x;
		if (x/itemsize != num || size < 0) {
			PyErr_SetString(StructError,
					"total struct size too long");
			return -1;
		}
	}

	/* check for overflow */
	if ((len + 1) > (PY_SSIZE_T_MAX / sizeof(formatcode))) {
		PyErr_NoMemory();
		return -1;
	}

	self->s_size = size;
	self->s_len = len;
	codes = (formatcode *) PyMem_MALLOC((len + 1) * sizeof(formatcode));
	if (codes == NULL) {
		PyErr_NoMemory();
		return -1;
	}
	self->s_codes = codes;

	s = fmt;
	size = 0;
	while ((c = *s++) != '\0') {
		if (isspace(Py_CHARMASK(c)))
			continue;
		if ('0' <= c && c <= '9') {
			num = c - '0';
			while ('0' <= (c = *s++) && c <= '9')
				num = num*10 + (c - '0');
			if (c == '\0')
				break;
		}
		else
			num = 1;

		e = getentry(c, f);

		size = align(size, c, e);
		if (c == 's' || c == 'p') {
			codes->offset = size;
			codes->size = num;
			codes->fmtdef = e;
			codes++;
			size += num;
		} else if (c == 'x') {
			size += num;
		} else {
			while (--num >= 0) {
				codes->offset = size;
				codes->size = e->size;
				codes->fmtdef = e;
				codes++;
				size += e->size;
			}
		}
	}
	codes->fmtdef = NULL;
	codes->offset = size;
	codes->size = 0;

	return 0;
}

static PyObject *
s_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
	PyObject *self;

	assert(type != NULL && type->tp_alloc != NULL);

	self = type->tp_alloc(type, 0);
	if (self != NULL) {
		PyStructObject *s = (PyStructObject*)self;
		Py_INCREF(Py_None);
		s->s_format = Py_None;
		s->s_codes = NULL;
		s->s_size = -1;
		s->s_len = -1;
	}
	return self;
}

static int
s_init(PyObject *self, PyObject *args, PyObject *kwds)
{
	PyStructObject *soself = (PyStructObject *)self;
	PyObject *o_format = NULL;
	int ret = 0;
	static char *kwlist[] = {"format", 0};

	assert(PyStruct_Check(self));

	if (!PyArg_ParseTupleAndKeywords(args, kwds, "S:Struct", kwlist,
					 &o_format))
		return -1;

	Py_INCREF(o_format);
	Py_CLEAR(soself->s_format);
	soself->s_format = o_format;

	ret = prepare_s(soself);
	return ret;
}

static void
s_dealloc(PyStructObject *s)
{
	if (s->weakreflist != NULL)
		PyObject_ClearWeakRefs((PyObject *)s);
	if (s->s_codes != NULL) {
		PyMem_FREE(s->s_codes);
	}
	Py_XDECREF(s->s_format);
	Py_TYPE(s)->tp_free((PyObject *)s);
}

static PyObject *
s_unpack_internal(PyStructObject *soself, char *startfrom) {
	formatcode *code;
	Py_ssize_t i = 0;
	PyObject *result = PyTuple_New(soself->s_len);
	if (result == NULL)
		return NULL;

	for (code = soself->s_codes; code->fmtdef != NULL; code++) {
		PyObject *v;
		const formatdef *e = code->fmtdef;
		const char *res = startfrom + code->offset;
		if (e->format == 's') {
			v = PyString_FromStringAndSize(res, code->size);
		} else if (e->format == 'p') {
			Py_ssize_t n = *(unsigned char*)res;
			if (n >= code->size)
				n = code->size - 1;
			v = PyString_FromStringAndSize(res + 1, n);
		} else {
			v = e->unpack(res, e);
		}
		if (v == NULL)
			goto fail;
		PyTuple_SET_ITEM(result, i++, v);
	}

	return result;
fail:
	Py_DECREF(result);
	return NULL;
}


static PyObject *
s_unpack(PyObject *self, PyObject *inputstr)
{
	char *start;
	Py_ssize_t len;
	PyObject *args=NULL, *result;
	PyStructObject *soself = (PyStructObject *)self;
	assert(PyStruct_Check(self));
	assert(soself->s_codes != NULL);
	if (inputstr == NULL)
		goto fail;
	if (PyString_Check(inputstr) &&
		PyString_GET_SIZE(inputstr) == soself->s_size) {
			return s_unpack_internal(soself, PyString_AS_STRING(inputstr));
	}
	args = PyTuple_Pack(1, inputstr);
	if (args == NULL)
		return NULL;
	if (!PyArg_ParseTuple(args, "s#:unpack", &start, &len))
		goto fail;
	if (soself->s_size != len)
		goto fail;
	result = s_unpack_internal(soself, start);
	Py_DECREF(args);
	return result;

fail:
	Py_XDECREF(args);
	PyErr_Format(StructError,
		"unpack requires a string argument of length %zd",
		soself->s_size);
	return NULL;
}

static PyObject *
s_unpack_from(PyObject *self, PyObject *args, PyObject *kwds)
{
	static char *kwlist[] = {"buffer", "offset", 0};
#if (PY_VERSION_HEX < 0x02050000)
	static char *fmt = "z#|i:unpack_from";
#else
	static char *fmt = "z#|n:unpack_from";
#endif
	Py_ssize_t buffer_len = 0, offset = 0;
	char *buffer = NULL;
	PyStructObject *soself = (PyStructObject *)self;
	assert(PyStruct_Check(self));
	assert(soself->s_codes != NULL);

	if (!PyArg_ParseTupleAndKeywords(args, kwds, fmt, kwlist,
					 &buffer, &buffer_len, &offset))
		return NULL;

	if (buffer == NULL) {
		PyErr_Format(StructError,
			"unpack_from requires a buffer argument");
		return NULL;
	}

	if (offset < 0)
		offset += buffer_len;

	if (offset < 0 || (buffer_len - offset) < soself->s_size) {
		PyErr_Format(StructError,
			"unpack_from requires a buffer of at least %zd bytes",
			soself->s_size);
		return NULL;
	}
	return s_unpack_internal(soself, buffer + offset);
}


/*
 * Guts of the pack function.
 *
 * Takes a struct object, a tuple of arguments, and offset in that tuple of
 * argument for where to start processing the arguments for packing, and a
 * character buffer for writing the packed string.  The caller must insure
 * that the buffer may contain the required length for packing the arguments.
 * 0 is returned on success, 1 is returned if there is an error.
 *
 */
static int
s_pack_internal(PyStructObject *soself, PyObject *args, int offset, char* buf)
{
	formatcode *code;
	/* XXX(nnorwitz): why does i need to be a local?  can we use
	   the offset parameter or do we need the wider width? */
	Py_ssize_t i;

	memset(buf, '\0', soself->s_size);
	i = offset;
	for (code = soself->s_codes; code->fmtdef != NULL; code++) {
		Py_ssize_t n;
		PyObject *v = PyTuple_GET_ITEM(args, i++);
		const formatdef *e = code->fmtdef;
		char *res = buf + code->offset;
		if (e->format == 's') {
			if (!PyString_Check(v)) {
				if (!PyObject_CheckReadBuffer(v))
				{
					PyErr_SetString(StructError,
							"argument for 's' must "
							"be a string or a buffer");
					return -1;
				}
				else
				{
					const void *buf;
					Py_ssize_t len;
					if (PyObject_AsReadBuffer(v, &buf, &len))
						return -1;

					if (len > code->size)
						len = code->size;
					if (len > 0)
						memcpy(res, buf, len);
				}
			}
			else
			{
				n = PyString_GET_SIZE(v);
				if (n > code->size)
					n = code->size;
				if (n > 0)
					memcpy(res, PyString_AS_STRING(v), n);
			}
		} else if (e->format == 'p') {
			if (!PyString_Check(v)) {
				PyErr_SetString(StructError,
						"argument for 'p' must "
						"be a string");
				return -1;
			}
			n = PyString_GET_SIZE(v);
			if (n > (code->size - 1))
				n = code->size - 1;
			if (n > 0)
				memcpy(res + 1, PyString_AS_STRING(v), n);
			if (n > 255)
				n = 255;
			*res = Py_SAFE_DOWNCAST(n, Py_ssize_t, unsigned char);
		} else if (e->pack(res, v, e) < 0) {
			if (strchr(integer_codes, e->format) != NULL &&
			    PyErr_ExceptionMatches(PyExc_OverflowError))
				PyErr_Format(StructError,
					     "integer out of range for "
					     "'%c' format code",
					     e->format);
			return -1;
		}
	}

	/* Success */
	return 0;
}


static PyObject *
s_pack(PyObject *self, PyObject *args)
{
	PyStructObject *soself;
	PyObject *result;

	/* Validate arguments. */
	soself = (PyStructObject *)self;
	assert(PyStruct_Check(self));
	assert(soself->s_codes != NULL);
	if (PyTuple_GET_SIZE(args) != soself->s_len)
	{
		PyErr_Format(StructError,
			"pack requires exactly %zd arguments", soself->s_len);
		return NULL;
	}

	/* Allocate a new string */
	result = PyString_FromStringAndSize((char *)NULL, soself->s_size);
	if (result == NULL)
		return NULL;

	/* Call the guts */
	if ( s_pack_internal(soself, args, 0, PyString_AS_STRING(result)) != 0 ) {
		Py_DECREF(result);
		return NULL;
	}

	return result;
}


static PyObject *
s_pack_into(PyObject *self, PyObject *args)
{
	PyStructObject *soself;
	char *buffer;
	Py_ssize_t buffer_len, offset;

	/* Validate arguments.  +1 is for the first arg as buffer. */
	soself = (PyStructObject *)self;
	assert(PyStruct_Check(self));
	assert(soself->s_codes != NULL);
	if (PyTuple_GET_SIZE(args) != (soself->s_len + 2))
	{
		PyErr_Format(StructError,
			     "pack_into requires exactly %zd arguments",
			     (soself->s_len + 2));
		return NULL;
	}

	/* Extract a writable memory buffer from the first argument */
	if ( PyObject_AsWriteBuffer(PyTuple_GET_ITEM(args, 0),
								(void**)&buffer, &buffer_len) == -1 ) {
		return NULL;
	}
	assert( buffer_len >= 0 );

	/* Extract the offset from the first argument */
	offset = PyInt_AsSsize_t(PyTuple_GET_ITEM(args, 1));
	if (offset == -1 && PyErr_Occurred())
		return NULL;

	/* Support negative offsets. */
	if (offset < 0)
		offset += buffer_len;

	/* Check boundaries */
	if (offset < 0 || (buffer_len - offset) < soself->s_size) {
		PyErr_Format(StructError,
			     "pack_into requires a buffer of at least %zd bytes",
			     soself->s_size);
		return NULL;
	}

	/* Call the guts */
	if ( s_pack_internal(soself, args, 2, buffer + offset) != 0 ) {
		return NULL;
	}

	Py_RETURN_NONE;
}

static PyObject *
s_get_format(PyStructObject *self, void *unused)
{
	Py_INCREF(self->s_format);
	return self->s_format;
}

static PyObject *
s_get_size(PyStructObject *self, void *unused)
{
    return PyInt_FromSsize_t(self->s_size);
}

/* ---- Standalone functions  ---- */

#define MAXCACHE 100
static PyObject *cache = NULL;

static PyObject *
cache_struct(PyObject *fmt)
{
	PyObject * s_object;

	if (cache == NULL) {
		cache = PyDict_New();
		if (cache == NULL)
			return NULL;
	}

	s_object = PyDict_GetItem(cache, fmt);
	if (s_object != NULL) {
		Py_INCREF(s_object);
		return s_object;
	}

	s_object = PyObject_CallFunctionObjArgs((PyObject *)(&PyStructType), fmt, NULL);
	if (s_object != NULL) {
		if (PyDict_Size(cache) >= MAXCACHE)
			PyDict_Clear(cache);
		/* Attempt to cache the result */
		if (PyDict_SetItem(cache, fmt, s_object) == -1)
			PyErr_Clear();
	}
	return s_object;
}

PyDoc_STRVAR(clearcache_doc,
"Clear the internal cache.");

static PyObject *
clearcache(PyObject *self)
{
	Py_CLEAR(cache);
	Py_RETURN_NONE;
}

PyDoc_STRVAR(calcsize_doc,
"Return size of C struct described by format string fmt.");

static PyObject *
calcsize(PyObject *self, PyObject *fmt)
{
	Py_ssize_t n;
	PyObject *s_object = cache_struct(fmt);
	if (s_object == NULL)
		return NULL;
	n = ((PyStructObject *)s_object)->s_size;
	Py_DECREF(s_object);
    	return PyInt_FromSsize_t(n);
}

PyDoc_STRVAR(pack_doc,
"Return string containing values v1, v2, ... packed according to fmt.");

static PyObject *
pack(PyObject *self, PyObject *args)
{
	PyObject *s_object, *fmt, *newargs, *result;
	Py_ssize_t n = PyTuple_GET_SIZE(args);

	if (n == 0) {
		PyErr_SetString(PyExc_TypeError, "missing format argument");
		return NULL;
	}
	fmt = PyTuple_GET_ITEM(args, 0);
	newargs = PyTuple_GetSlice(args, 1, n);
	if (newargs == NULL)
		return NULL;

	s_object = cache_struct(fmt);
	if (s_object == NULL) {
		Py_DECREF(newargs);
		return NULL;
	}
    	result = s_pack(s_object, newargs);
	Py_DECREF(newargs);
	Py_DECREF(s_object);
	return result;
}

PyDoc_STRVAR(pack_into_doc,
"Pack the values v1, v2, ... according to fmt.\n\
Write the packed bytes into the writable buffer buf starting at offset.");

static PyObject *
pack_into(PyObject *self, PyObject *args)
{
	PyObject *s_object, *fmt, *newargs, *result;
	Py_ssize_t n = PyTuple_GET_SIZE(args);

	if (n == 0) {
		PyErr_SetString(PyExc_TypeError, "missing format argument");
		return NULL;
	}
	fmt = PyTuple_GET_ITEM(args, 0);
	newargs = PyTuple_GetSlice(args, 1, n);
	if (newargs == NULL)
		return NULL;

	s_object = cache_struct(fmt);
	if (s_object == NULL) {
		Py_DECREF(newargs);
		return NULL;
	}
    	result = s_pack_into(s_object, newargs);
	Py_DECREF(newargs);
	Py_DECREF(s_object);
	return result;
}

PyDoc_STRVAR(unpack_doc,
"Unpack the string containing packed C structure data, according to fmt.\n\
Requires len(string) == calcsize(fmt).");

static PyObject *
unpack(PyObject *self, PyObject *args)
{
	PyObject *s_object, *fmt, *inputstr, *result;

	if (!PyArg_UnpackTuple(args, "unpack", 2, 2, &fmt, &inputstr))
		return NULL;

	s_object = cache_struct(fmt);
	if (s_object == NULL)
		return NULL;
    	result = s_unpack(s_object, inputstr);
	Py_DECREF(s_object);
	return result;
}

PyDoc_STRVAR(unpack_from_doc,
"Unpack the buffer, containing packed C structure data, according to\n\
fmt, starting at offset. Requires len(buffer[offset:]) >= calcsize(fmt).");

static PyObject *
unpack_from(PyObject *self, PyObject *args, PyObject *kwds)
{
	PyObject *s_object, *fmt, *newargs, *result;
	Py_ssize_t n = PyTuple_GET_SIZE(args);

	if (n == 0) {
		PyErr_SetString(PyExc_TypeError, "missing format argument");
		return NULL;
	}
	fmt = PyTuple_GET_ITEM(args, 0);
	newargs = PyTuple_GetSlice(args, 1, n);
	if (newargs == NULL)
		return NULL;

	s_object = cache_struct(fmt);
	if (s_object == NULL) {
		Py_DECREF(newargs);
		return NULL;
	}
    	result = s_unpack_from(s_object, newargs, kwds);
	Py_DECREF(newargs);
	Py_DECREF(s_object);
	return result;
}

static struct PyMethodDef module_functions[] = {
	{"_clearcache",	(PyCFunction)clearcache,	METH_NOARGS, 	clearcache_doc},
	{"calcsize",	calcsize,	METH_O, 	calcsize_doc},
	{"pack",	pack,		METH_VARARGS, 	pack_doc},
	{"pack_into",	pack_into,	METH_VARARGS, 	pack_into_doc},
	{"unpack",	unpack,       	METH_VARARGS, 	unpack_doc},
	{"unpack_from",	(PyCFunction)unpack_from, 	
			METH_VARARGS|METH_KEYWORDS, 	unpack_from_doc},
	{NULL,	 NULL}		/* sentinel */
};


/* Module initialization */

PyDoc_STRVAR(module_doc,
"Functions to convert between Python values and C structs represented\n\
as Python strings. It uses format strings (explained below) as compact\n\
descriptions of the lay-out of the C structs and the intended conversion\n\
to/from Python values.\n\
\n\
The remaining chars indicate types of args and must match exactly;\n\
these can be preceded by a decimal repeat count:\n\
  x: pad byte (no data); c:char; b:signed byte; B:unsigned byte;\n\
  ?: _Bool (requires C99; if not available, char is used instead)\n\
  h:short; H:unsigned short; i:int; I:unsigned int;\n\
  l:long; L:unsigned long; f:float; d:double.\n\
Special cases (preceding decimal count indicates length):\n\
  s:string (array of char); p: pascal string (with count byte).\n\
Special case (only available in native format):\n\
  P:an integer type that is wide enough to hold a pointer.\n\
Special case (not in native mode unless 'long long' in platform C):\n\
  q:long long; Q:unsigned long long\n\
Whitespace between formats is ignored.\n\
\n\
The variable struct.error is an exception raised on errors.\n");

PyMODINIT_FUNC
init_pvt_struct(void)
{
	PyObject *ver, *m;

	ver = PyString_FromString("0.2");
	if (ver == NULL)
		return;

	m = Py_InitModule3("_pvt_struct", module_functions, module_doc);
	if (m == NULL)
		return;

	Py_TYPE(&PyStructType) = &PyType_Type;
	if (PyType_Ready(&PyStructType) < 0)
		return;

	/* This speed trick can't be used until overflow masking goes
	   away, because native endian always raises exceptions
	   instead of overflow masking. */

	/* Add some symbolic constants to the module */
	if (StructError == NULL) {
		StructError = PyErr_NewException("pycuda._pvt_struct.error", NULL, NULL);
		if (StructError == NULL)
			return;
	}

	Py_INCREF(StructError);
	PyModule_AddObject(m, "error", StructError);

	Py_INCREF((PyObject*)&PyStructType);
	PyModule_AddObject(m, "Struct", (PyObject*)&PyStructType);

	PyModule_AddObject(m, "__version__", ver);

	PyModule_AddIntConstant(m, "_PY_STRUCT_RANGE_CHECKING", 1);
#ifdef PY_STRUCT_FLOAT_COERCE
	PyModule_AddIntConstant(m, "_PY_STRUCT_FLOAT_COERCE", 1);
#endif

}

// vim: noexpandtab:sw=8
