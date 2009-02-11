GL Interoperability
===================

.. note::

    This functionality is scheduled for release 0.93 and only available in
    source-built versions of PyCuda's git tree.

.. module:: pycuda.gl

.. function :: init()
.. function :: make_context(dev, flags=0)

.. class :: BufferObject(bufobj)

    .. method :: unregister()
    .. method :: handle()
    .. method :: map()
    
.. class :: BufferObjectMapping

    .. method :: unmap()
    .. method :: device_ptr()
    .. method :: size()

Automatic Initialization
------------------------

.. module:: pycuda.gl.autoinit

.. data:: device
.. data:: context
