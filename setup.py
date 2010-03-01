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




def search_on_path(filenames):
    """Find file on system path."""
    # http://aspn.activestate.com/ASPN/Cookbook/Python/Recipe/52224

    from os.path import exists, join, abspath
    from os import pathsep, environ

    search_path = environ["PATH"]

    file_found = 0
    paths = search_path.split(pathsep)
    for path in paths:
        for filename in filenames:
            if exists(join(path, filename)):
                return abspath(join(path, filename))




# verification ----------------------------------------------------------------
def verify_path(description, paths, names, extensions, subpaths=['/'],
        prefixes=[], maybe_ok=False):
    try:
        from os.path import exists

        defaultname = names[0] + extensions[0]
        prefixes.append("")
        looked_where = []

        for path in paths:
            for subpath in subpaths:
                for prefix in prefixes:
                    for name in names:
                        for extension in extensions:
                            print path, subpath, prefix, name, extension
                            filename = path + subpath + prefix + name + extension

                            looked_where.append(filename)

                            if exists(filename):
                                return
        print "*** Cannot find %s. Checked locations:" % description
        for path in looked_where:
            print "   %s" % path

        if maybe_ok:
            print "*** Note that this may not be a problem as this " \
                    "component is often installed system-wide."
    except:
        print "*** Error occurred in plausibility checking for path of %s." \
                % description




def verify_siteconfig(sc_vars):
    LIB_EXTS = ['.so', '.dylib', '.lib']
    LIB_PREFIXES = ['lib']
    APP_EXTS = ['', '.exe']
    warn_prefix = '!! Warning: '

    # BOOST_INC_DIR/boost/python.hpp
    if 'BOOST_INC_DIR' in sc_vars:
        verify_path (
            description="Boost headers",
            paths=sc_vars['BOOST_INC_DIR'],
            subpaths=['/boost/'],
            names=['python'],
            extensions=['.hpp']
            );
    else:
        print warn_prefix + 'BOOST_INC_DIR is not set, should be something like "/path/to/boost/include/boost-1_39".'

    # BOOST_LIB_DIR/(lib)?BOOST_PYTHON_LIBNAME(.so|.dylib|?Windows?)
    if 'BOOST_LIB_DIR' not in sc_vars:
        print warn_prefix + 'BOOST_LIB_DIR is not set, should be like BOOST_INC_DIR but with "/lib" instead of "/include/boost-1_39".'

    if 'BOOST_PYTHON_LIBNAME' in sc_vars:
        verify_path (
            description="Boost Python library",
            paths=sc_vars['BOOST_LIB_DIR'],
            names=sc_vars['BOOST_PYTHON_LIBNAME'],
            extensions=LIB_EXTS,
            prefixes=LIB_PREFIXES
            )
    else:
        print warn_prefix + 'BOOST_PYTHON_LIBNAME is not set, should be something like "boost_python-*-mt".'

    # BOOST_LIB_DIR/(lib)?BOOST_THREAD_LIBNAME(.so|.dylib|?Windows?)
    if 'BOOST_THREAD_LIBNAME' in sc_vars:
        verify_path(
            description="Boost Thread library",
            paths=sc_vars['BOOST_LIB_DIR'],
            names=sc_vars['BOOST_THREAD_LIBNAME'],
            extensions=LIB_EXTS,
            prefixes=LIB_PREFIXES
            )
    else:
        print warn_prefix + 'BOOST_THREAD_LIBNAME is not set, should be something like "boost_thread-*-mt".'

    # CUDA_ROOT/bin/nvcc(.exe)?
    if 'CUDA_ROOT' in sc_vars:
        verify_path(
            description="CUDA toolkit",
            paths=[sc_vars['CUDA_ROOT']],
            subpaths=['/bin/'],
            names=['nvcc'],
            extensions=APP_EXTS,
            )
    else:
        print warn_prefix + 'CUDA_ROOT is not set, should point to the nVidia CUDA Toolkit.'

    # CUDA_INC_DIR/cuda.h
    if sc_vars.has_key('CUDA_INC_DIR'):
        verify_path (
            description="CUDA include directory",
            paths=sc_vars['CUDA_INC_DIR'],
            names=['cuda'],
            extensions=['.h'],
            )
    else:
        print warn_prefix + 'CUDA_INC_DIR is not set, should be something like CUDA_ROOT + "/include".'

    # CUDADRV_LIB_DIR=(lib)?CUDADRV_LIBNAME(.so|.dylib|?Windows?)
    if not sc_vars.has_key('CUDADRV_LIB_DIR'):
        print warn_prefix + 'CUDADRV_LIB_DIR is not set, should be something like CUDA_ROOT + "/lib".'

    if 'CUDADRV_LIBNAME' in sc_vars:
        verify_path (
            description="CUDA driver library",
            paths=sc_vars['CUDADRV_LIB_DIR'],
            names=sc_vars['CUDADRV_LIBNAME'],
            extensions=LIB_EXTS,
            prefixes=LIB_PREFIXES,
            maybe_ok=True,
            )
    else:
        print warn_prefix + 'CUDADRV_LIBNAME is not set, should most likely be "cuda".'




# main functionality ----------------------------------------------------------
def main():
    import glob
    from aksetup_helper import hack_distutils, get_config, setup, \
            NumpyExtension, Extension

    hack_distutils()
    conf = get_config(get_config_schema())

    LIBRARY_DIRS = conf["BOOST_LIB_DIR"]
    LIBRARIES = conf["BOOST_PYTHON_LIBNAME"] + conf["BOOST_THREAD_LIBNAME"]

    from os.path import dirname, join, normpath

    if conf["CUDA_ROOT"] is None:
        nvcc_path = search_on_path(["nvcc", "nvcc.exe"])
        if nvcc_path is None:
            print "*** CUDA_ROOT not set, and nvcc not in path. Giving up."
            import sys
            sys.exit(1)

        conf["CUDA_ROOT"] = normpath(join(dirname(nvcc_path), ".."))

    if conf["CUDA_INC_DIR"] is None:
        conf["CUDA_INC_DIR"] = [join(conf["CUDA_ROOT"], "include")]
    if not conf["CUDADRV_LIB_DIR"]:
        conf["CUDADRV_LIB_DIR"] = [join(conf["CUDA_ROOT"], "lib")]

    verify_siteconfig(conf)

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
        # also, default to 32-bit build, since there doesn't appear to be a
        # 64-bit CUDA on Mac yet.
        if "-arch" not in conf["CXXFLAGS"]:
            conf["CXXFLAGS"].extend(['-arch', 'i386', '-m32'])
        if "-arch" not in conf["LDFLAGS"]:
            conf["LDFLAGS"].extend(['-arch', 'i386', '-m32'])

    ext_kwargs = dict()

    extra_sources = []
    if conf["CUDA_ENABLE_GL"]:
        extra_sources.append("src/wrapper/wrap_cudagl.cpp")
        EXTRA_DEFINES["HAVE_GL"] = 1

    ver_dic = {}
    execfile("pycuda/__init__.py", ver_dic)

    setup(name="pycuda",
            # metadata
            version=ver_dic["VERSION_TEXT"],
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
              your disposal, if you wish. It also includes code for
              interoperability with OpenGL.

            * Automatic Error Checking. All CUDA errors are automatically
              translated into Python exceptions.

            * Speed. PyCUDA's base layer is written in C++, so all the niceties
              above are virtually free.

            * Helpful `Documentation <http://documen.tician.de/pycuda>`_ and a
              `Wiki <http://wiki.tiker.net/PyCuda>`_.

            Relatedly, like-minded computing goodness for `OpenCL <http://khronos.org>`_
            is provided by PyCUDA's sister project `PyOpenCL <http://pypi.python.org/pypi/pyopencl>`_.
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

            install_requires=[
                "pytools>=8",
                "py>=1.0.0b7"
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
                Extension("_pvt_struct",
                    ["src/wrapper/_pycuda_struct.c"],
                    )],

            data_files=[
                ("include/pycuda", glob.glob("src/cuda/*.hpp"))
                ],
            )




if __name__ == '__main__':
    main()
