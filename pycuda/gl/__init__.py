import pycuda._driver as _drv

if not _drv.have_gl_ext(): 
    raise ImportError("PyCUDA was compiled without GL extension support")

init = _drv.gl_init
make_context = _drv.make_gl_context
BufferObject = _drv.BufferObject
BufferObjectMapping = _drv.BufferObjectMapping
