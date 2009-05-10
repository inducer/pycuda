#!/usr/bin/env python
# -*- coding: latin-1 -*-



def get_config_schema():
    from aksetup_helper import ConfigSchema, Option, \
            IncludeDir, LibraryDir, Libraries, BoostLibraries, \
            Switch, StringListOption, make_boost_base_options

    return ConfigSchema(make_boost_base_options() + [
        BoostLibraries("python"),
        BoostLibraries("thread"),

        Switch("CUDA_TRACE", False, "Enable CUDA API tracing"),
        Option("CUDA_ROOT", help="Path to the CUDA toolkit"),
        IncludeDir("CUDA", None),

        Switch("CUDA_ENABLE_GL", False, "Enable CUDA GL interoperability"),

        LibraryDir("CUDADRV", []),
        Libraries("CUDADRV", ["cuda"]),

        StringListOption("CXXFLAGS", [], 
            help="Any extra C++ compiler options to include"),
        StringListOption("LDFLAGS", [], 
            help="Any extra linker options to include"),
        ])




def search_on_path(filename):
    """Find file on system path."""
    # http://aspn.activestate.com/ASPN/Cookbook/Python/Recipe/52224

    from os.path import exists, join, abspath
    from os import pathsep, environ

    search_path = environ["PATH"]
    #print "*", search_path

    file_found = 0
    paths = search_path.split(pathsep)
    for path in paths:
        #print path
        if exists(join(path, filename)):
             file_found = 1
             break
    if file_found:
        return abspath(join(path, filename))
    else:
        return None




def main():
    import glob
    from aksetup_helper import hack_distutils, get_config, setup, \
            NumpyExtension

    hack_distutils()
    conf = get_config(get_config_schema())

    LIBRARY_DIRS = conf["BOOST_LIB_DIR"]
    LIBRARIES = conf["BOOST_PYTHON_LIBNAME"] + conf["BOOST_THREAD_LIBNAME"]

    from os.path import dirname, join, normpath

    if conf["CUDA_ROOT"] is None:
        nvcc_path = search_on_path("nvcc")
        if nvcc_path is None:
            print "*** CUDA_ROOT not set, and nvcc not in path. Giving up."
            import sys
            sys.exit(1)
            
        conf["CUDA_ROOT"] = normpath(join(dirname(nvcc_path), ".."))

    if conf["CUDA_INC_DIR"] is None:
        conf["CUDA_INC_DIR"] = [join(conf["CUDA_ROOT"], "include")]
    if not conf["CUDADRV_LIB_DIR"]:
        conf["CUDADRV_LIB_DIR"] = [join(conf["CUDA_ROOT"], "lib")]

    EXTRA_DEFINES = { }
    EXTRA_INCLUDE_DIRS = []
    EXTRA_LIBRARY_DIRS = []
    EXTRA_LIBRARIES = []

    if conf["CUDA_TRACE"]:
        EXTRA_DEFINES["CUDAPP_TRACE_CUDA"] = 1

    INCLUDE_DIRS = ['src/cpp'] + conf["BOOST_INC_DIR"] + conf["CUDA_INC_DIR"]
    conf["USE_CUDA"] = True

    import sys

    if 'darwin' in sys.platform:
        # prevent from building ppc since cuda on OS X is not compiled for ppc
        if "-arch" not in conf["CXXFLAGS"]:
            conf["CXXFLAGS"].extend(['-arch', 'i386'])
        if "-arch" not in conf["LDFLAGS"]:
            conf["LDFLAGS"].extend(['-arch', 'i386'])

    ext_kwargs = dict()

    extra_sources = []
    if conf["CUDA_ENABLE_GL"]:
        extra_sources.append("src/wrapper/wrap_cudagl.cpp")
        EXTRA_DEFINES["HAVE_GL"] = 1

    setup(name="pycuda",
            # metadata
            version="0.93rc1",
            description="Python wrapper for Nvidia CUDA",
            long_description="""
            PyCUDA lets you access `Nvidia <http://nvidia.com>`_'s `CUDA
            <http://nvidia.com/cuda/>`_ parallel computation API from Python.
            Several wrappers of the CUDA API already exist-so what's so special
            about PyCUDA?

            * Object cleanup tied to lifetime of objects. This idiom, often
              called
              `RAII <http://en.wikipedia.org/wiki/Resource_Acquisition_Is_Initialization>`_
              in C++, makes it much easier to write correct, leak- and
              crash-free code. PyCUDA knows about dependencies, too, so (for
              example) it won't detach from a context before all memory
              allocated in it is also freed.

            * Convenience. Abstractions like pycuda.driver.SourceModule and
              pycuda.gpuarray.GPUArray make CUDA programming even more
              convenient than with Nvidia's C-based runtime.

            * Completeness. PyCUDA puts the full power of CUDA's driver API at
              your disposal, if you wish.

            * Automatic Error Checking. All CUDA errors are automatically
              translated into Python exceptions.

            * Speed. PyCUDA's base layer is written in C++, so all the niceties
              above are virtually free.

            * Helpful `Documentation <http://documen.tician.de/pycuda>`_.
            """,
            author=u"Andreas Kloeckner",
            author_email="inform@tiker.net",
            license = "MIT",
            url="http://mathema.tician.de/software/pycuda",
            classifiers=[
              'Environment :: Console',
              'Development Status :: 5 - Production/Stable',
              'Intended Audience :: Developers',
              'Intended Audience :: Other Audience',
              'Intended Audience :: Science/Research',
              'License :: OSI Approved :: MIT License',
              'Natural Language :: English',
              'Programming Language :: C++',
              'Programming Language :: Python',
              'Topic :: Scientific/Engineering',
              'Topic :: Scientific/Engineering :: Mathematics',
              'Topic :: Scientific/Engineering :: Physics',
              'Topic :: Scientific/Engineering :: Visualization',
              ],

            # build info
            packages=["pycuda", "pycuda.gl"],
            zip_safe=False,

            install_requires=[
                "pytools>=8",
                ],

            ext_package="pycuda",
            ext_modules=[
                NumpyExtension("_driver", 
                    [
                        "src/cpp/cuda.cpp", 
                        "src/cpp/bitlog.cpp", 
                        "src/wrapper/wrap_cudadrv.cpp", 
                        "src/wrapper/mempool.cpp", 
                        ]+extra_sources, 
                    include_dirs=INCLUDE_DIRS + EXTRA_INCLUDE_DIRS,
                    library_dirs=LIBRARY_DIRS + conf["CUDADRV_LIB_DIR"],
                    libraries=LIBRARIES + conf["CUDADRV_LIBNAME"],
                    define_macros=list(EXTRA_DEFINES.iteritems()),
                    extra_compile_args=conf["CXXFLAGS"],
                    extra_link_args=conf["LDFLAGS"],
                    ),
                ],
                
            data_files=[
                ("include/cuda", glob.glob("src/cuda/*.hpp"))
                ],
            )




if __name__ == '__main__':
    main()
