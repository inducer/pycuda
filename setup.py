#!/usr/bin/env python
# -*- coding: latin-1 -*-


def get_config_schema():
    from aksetup_helper import ConfigSchema, Option, \
            IncludeDir, LibraryDir, Libraries, BoostLibraries, \
            Switch, StringListOption, make_boost_base_options

    return ConfigSchema(make_boost_base_options() + [
        Switch("USE_SHIPPED_BOOST", True, "Use included Boost library"),

        BoostLibraries("python"),
        BoostLibraries("thread"),

        Switch("CUDA_TRACE", False, "Enable CUDA API tracing"),
        Option("CUDA_ROOT", help="Path to the CUDA toolkit"),
        Option("CUDA_PRETEND_VERSION", help="Assumed CUDA version, in the form 3010 for 3.1."),
        IncludeDir("CUDA", None),

        Switch("CUDA_ENABLE_GL", False, "Enable CUDA GL interoperability"),
        Switch("CUDA_ENABLE_CURAND", True, "Enable CURAND library"),

        LibraryDir("CUDADRV", []),
        Libraries("CUDADRV", ["cuda"]),

        LibraryDir("CUDART", []),
        Libraries("CUDART", ["cudart"]),

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

        prefixes.append("")
        looked_where = []

        for path in paths:
            for subpath in subpaths:
                for prefix in prefixes:
                    for name in names:
                        for extension in extensions:
                            print(path, subpath, prefix, name, extension)
                            filename = path + subpath + prefix + name + extension

                            looked_where.append(filename)

                            if exists(filename):
                                return
        print("*** Cannot find %s. Checked locations:" % description)
        for path in looked_where:
            print("   %s" % path)

        if maybe_ok:
            print("*** Note that this may not be a problem as this "
                    "component is often installed system-wide.")
    except:
        print("*** Error occurred in plausibility checking for path of %s."
                % description)




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
        print(warn_prefix + 'BOOST_INC_DIR is not set, should be something like '
                '"/path/to/boost/include/boost-1_39".')

    # BOOST_LIB_DIR/(lib)?BOOST_PYTHON_LIBNAME(.so|.dylib|?Windows?)
    if 'BOOST_LIB_DIR' not in sc_vars:
        print(warn_prefix + 'BOOST_LIB_DIR is not set, should be '
                'like BOOST_INC_DIR but with "/lib" instead of '
                '"/include/boost-1_39".')

    if 'BOOST_PYTHON_LIBNAME' in sc_vars:
        verify_path (
            description="Boost Python library",
            paths=sc_vars['BOOST_LIB_DIR'],
            names=sc_vars['BOOST_PYTHON_LIBNAME'],
            extensions=LIB_EXTS,
            prefixes=LIB_PREFIXES
            )
    else:
        print(warn_prefix + 'BOOST_PYTHON_LIBNAME is not set, '
                'should be something like "boost_python-*-mt".')

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
        print(warn_prefix + 'BOOST_THREAD_LIBNAME is not set, '
                'should be something like "boost_thread-*-mt".')

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
        print(warn_prefix + 'CUDA_ROOT is not set, '
                'should point to the nVidia CUDA Toolkit.')

    # CUDA_INC_DIR/cuda.h
    if 'CUDA_INC_DIR' in sc_vars:
        verify_path (
            description="CUDA include directory",
            paths=sc_vars['CUDA_INC_DIR'],
            names=['cuda'],
            extensions=['.h'],
            )
    else:
        print(warn_prefix + 'CUDA_INC_DIR is not set, '
                'should be something like CUDA_ROOT + "/include".')

    # CUDADRV_LIB_DIR=(lib)?CUDADRV_LIBNAME(.so|.dylib|?Windows?)
    if not 'CUDADRV_LIB_DIR' in sc_vars:
        print(warn_prefix + 'CUDADRV_LIB_DIR is not set, should '
                'be something like CUDA_ROOT + "/lib".')

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
        print(warn_prefix + 'CUDADRV_LIBNAME is not set, should most likely be "cuda".')

    # CUDART_LIB_DIR=(lib)?CUDART_LIBNAME(.so|.dylib|?Windows?)
    if not 'CUDART_LIB_DIR' in sc_vars:
        # Since libcurand.so is distributed in the CUDA Runtime lib directory.
        # which is different from the CUDA Driver lib directory, we should check.
        print(warn_prefix + 'CUDART_LIB_DIR is not set, should '
                'be sometthing like CUDA_ROOT + "/lib".')

    if 'CUDART_LIBNAME' in sc_vars:
        verify_path (
            description="CUDA runtime library",
            paths=sc_vars['CUDART_LIB_DIR'],
            names=sc_vars['CUDART_LIBNAME'],
            extensions=LIB_EXTS,
            prefixes=LIB_PREFIXES,
            maybe_ok=True,
            )
    else:
        # This should be optional, since PyCUDA doesn't directly use the CUDART
        pass


# main functionality ----------------------------------------------------------
def main():
    import glob
    import sys
    from os.path import dirname, join, normpath

    from aksetup_helper import (hack_distutils, get_config, setup, \
            NumpyExtension, Extension, set_up_shipped_boost_if_requested,
            check_git_submodules)

    check_git_submodules()

    hack_distutils()
    conf = get_config(get_config_schema())
    EXTRA_SOURCES, EXTRA_DEFINES = set_up_shipped_boost_if_requested("pycuda", conf)

    EXTRA_DEFINES["PYGPU_PACKAGE"] = "pycuda"
    EXTRA_DEFINES["PYGPU_PYCUDA"] = "1"

    LIBRARY_DIRS = conf["BOOST_LIB_DIR"]
    LIBRARIES = conf["BOOST_PYTHON_LIBNAME"] + conf["BOOST_THREAD_LIBNAME"]

    if conf["CUDA_ROOT"] is None:
        nvcc_path = search_on_path(["nvcc", "nvcc.exe"])
        if nvcc_path is None:
            print("*** CUDA_ROOT not set, and nvcc not in path. Giving up.")
            sys.exit(1)

        conf["CUDA_ROOT"] = normpath(join(dirname(nvcc_path), ".."))

    if not conf["CUDA_INC_DIR"]:
        conf["CUDA_INC_DIR"] = [join(conf["CUDA_ROOT"], "include")]

    if not conf["CUDADRV_LIB_DIR"]:
        platform_bits = tuple.__itemsize__ * 8

        if platform_bits == 64 and 'darwin' not in sys.platform:
            lib_dir_name = "lib64"
        else:
            lib_dir_name = "lib"

        conf["CUDADRV_LIB_DIR"] = [join(conf["CUDA_ROOT"], lib_dir_name)]

    verify_siteconfig(conf)

    EXTRA_INCLUDE_DIRS = []
    EXTRA_LIBRARIES = []

    if conf["CUDA_TRACE"]:
        EXTRA_DEFINES["CUDAPP_TRACE_CUDA"] = 1

    if conf["CUDA_PRETEND_VERSION"]:
        EXTRA_DEFINES["CUDAPP_PRETEND_CUDA_VERSION"] = conf["CUDA_PRETEND_VERSION"]

    INCLUDE_DIRS = ['src/cpp'] + conf["BOOST_INC_DIR"] + conf["CUDA_INC_DIR"]
    conf["USE_CUDA"] = True


    if 'darwin' in sys.platform and sys.maxsize == 2147483647:
        # The Python interpreter is running in 32 bit mode on OS X
        if "-arch" not in conf["CXXFLAGS"]:
            conf["CXXFLAGS"].extend(['-arch', 'i386', '-m32'])
        if "-arch" not in conf["LDFLAGS"]:
            conf["LDFLAGS"].extend(['-arch', 'i386', '-m32'])

    if 'darwin' in sys.platform:
        # set path to Cuda dynamic libraries,
        # as a safe substitute for DYLD_LIBRARY_PATH
        for lib_dir in conf['CUDADRV_LIB_DIR']:
            conf["LDFLAGS"].extend(["-Xlinker", "-rpath", "-Xlinker", lib_dir])

    if conf["CUDA_ENABLE_GL"]:
        EXTRA_SOURCES.append("src/wrapper/wrap_cudagl.cpp")
        EXTRA_DEFINES["HAVE_GL"] = 1

    if conf["CUDA_ENABLE_CURAND"]:
        EXTRA_DEFINES["HAVE_CURAND"] = 1
        EXTRA_SOURCES.extend([
            "src/wrapper/wrap_curand.cpp"
            ])
        EXTRA_LIBRARIES.append("curand")

    ver_dic = {}
    exec(compile(open("pycuda/__init__.py").read(), "pycuda/__init__.py", 'exec'), ver_dic)

    try:
        from distutils.command.build_py import build_py_2to3 as build_py
    except ImportError:
        # 2.x
        from distutils.command.build_py import build_py

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
            author="Andreas Kloeckner",
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
            packages=["pycuda", "pycuda.gl", "pycuda.sparse", "pycuda.compyte"],

            install_requires=[
                "pytools>=2011.2",
                "pytest>=2",
                "decorator>=3.2.0"
                ],

            ext_package="pycuda",
            ext_modules=[
                NumpyExtension("_driver",
                    [
                        "src/cpp/cuda.cpp",
                        "src/cpp/bitlog.cpp",
                        "src/wrapper/wrap_cudadrv.cpp",
                        "src/wrapper/mempool.cpp",
                        ]+EXTRA_SOURCES,
                    include_dirs=INCLUDE_DIRS + EXTRA_INCLUDE_DIRS,
                    library_dirs=LIBRARY_DIRS + conf["CUDADRV_LIB_DIR"],
                    libraries=LIBRARIES + conf["CUDADRV_LIBNAME"] + EXTRA_LIBRARIES,
                    define_macros=list(EXTRA_DEFINES.items()),
                    extra_compile_args=conf["CXXFLAGS"],
                    extra_link_args=conf["LDFLAGS"],
                    ),
                Extension("_pvt_struct",
                    ["src/wrapper/_pycuda_struct.c"],
                    extra_compile_args=conf["CXXFLAGS"],
                    extra_link_args=conf["LDFLAGS"],
                    ),
                ],

            data_files=[
                ("include/pycuda", glob.glob("src/cuda/*.hpp"))
                ],

            # 2to3 invocation
            cmdclass={'build_py': build_py})


if __name__ == '__main__':
    main()
