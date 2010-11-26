import pycuda._driver as _drv

if not _drv.have_gl_ext(): 
    raise ImportError("PyCUDA was compiled without GL extension support")

init = _drv.gl_init
make_context = _drv.make_gl_context
map_flags = _drv.map_flags
target_flags = _drv.target_flags
BufferObject = _drv.BufferObject
BufferObjectMapping = _drv.BufferObjectMapping
RegisteredBuffer = _drv.RegisteredBuffer
RegisteredImage = _drv.RegisteredImage
RegisteredMapping = _drv.RegisteredMapping
