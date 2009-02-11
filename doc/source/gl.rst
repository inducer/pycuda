GL Interoperability
===================

.. note::

    This functionality is scheduled for release 0.93 and only available from 
    source-built versions from PyCuda's git tree.

.. module:: pycuda.gl

.. function :: init()
.. function :: make_context(dev, flags=0)

.. class :: BufferObject(bufobj)

    .. method :: unregister()
    .. method :: handle()
    .. method :: map()
    
.. class :: BufferObjectMapping(bufobj)

    .. method :: unmap()
    .. method :: device_ptr()
    .. method :: size()
