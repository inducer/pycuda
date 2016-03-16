#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import, print_function
from os.path import dirname, join, normpath


def search_on_path(filenames):
    """Find file on system path."""
    # http://aspn.activestate.com/ASPN/Cookbook/Python/Recipe/52224

    from os.path import exists, abspath
    from os import pathsep, environ

    search_path = environ["PATH"]

    paths = search_path.split(pathsep)
    for path in paths:
        for filename in filenames:
            if exists(join(path, filename)):
                return abspath(join(path, filename))


def get_config_schema():
    from aksetup_helper import ConfigSchema, Option, \
            IncludeDir, LibraryDir, Libraries, BoostLibraries, \
            Switch, StringListOption, make_boost_base_options

    nvcc_path = search_on_path(["nvcc", "nvcc.exe"])
    if nvcc_path is None:
        print("***************************************************************")
        print("*** WARNING: nvcc not in path.")
        print("*** May need to set CUDA_INC_DIR for installation to succeed.")
        print("***************************************************************")
        cuda_root_default = None
    else:
        cuda_root_default = normpath(join(dirname(nvcc_path), ".."))

    default_lib_dirs = [
        "${CUDA_ROOT}/lib", "${CUDA_ROOT}/lib64",
        # https://github.com/inducer/pycuda/issues/98
        "${CUDA_ROOT}/lib/stubs", "${CUDA_ROOT}/lib64/stubs",
        ]

    return ConfigSchema(make_boost_base_options() + [
        Switch("USE_SHIPPED_BOOST", True, "Use included Boost library"),

        BoostLibraries("python"),
        BoostLibraries("thread"),

        Switch("CUDA_TRACE", False, "Enable CUDA API tracing"),
        Option("CUDA_ROOT", default=cuda_root_default,
            help="Path to the CUDA toolkit"),
        Option("CUDA_PRETEND_VERSION",
            help="Assumed CUDA version, in the form 3010 for 3.1."),
        IncludeDir("CUDA", None),

        Switch("CUDA_ENABLE_GL", False, "Enable CUDA GL interoperability"),
        Switch("CUDA_ENABLE_CURAND", True, "Enable CURAND library"),

        LibraryDir("CUDADRV", default_lib_dirs),
        Libraries("CUDADRV", ["cuda"]),

        LibraryDir("CUDART", default_lib_dirs),
        Libraries("CUDART", ["cudart"]),

        LibraryDir("CURAND", default_lib_dirs),
        Libraries("CURAND", ["curand"]),

        StringListOption("CXXFLAGS", [],
            help="Any extra C++ compiler options to include"),
        StringListOption("LDFLAGS", [],
            help="Any extra linker options to include"),
        ])


def main():
    import sys

    from aksetup_helper import (hack_distutils, get_config, setup,
            NumpyExtension, set_up_shipped_boost_if_requested,
            check_git_submodules)

    check_git_submodules()

    hack_distutils()
    conf = get_config(get_config_schema())

    EXTRA_SOURCES, EXTRA_DEFINES = set_up_shipped_boost_if_requested("pycuda", conf)

    EXTRA_DEFINES["PYGPU_PACKAGE"] = "pycuda"
    EXTRA_DEFINES["PYGPU_PYCUDA"] = "1"

    LIBRARY_DIRS = conf["BOOST_LIB_DIR"] + conf["CUDADRV_LIB_DIR"]
    LIBRARIES = (conf["BOOST_PYTHON_LIBNAME"] + conf["BOOST_THREAD_LIBNAME"]
            + conf["CUDADRV_LIBNAME"])

    if not conf["CUDA_INC_DIR"] and conf["CUDA_ROOT"]:
        conf["CUDA_INC_DIR"] = [join(conf["CUDA_ROOT"], "include")]

    if conf["CUDA_TRACE"]:
        EXTRA_DEFINES["CUDAPP_TRACE_CUDA"] = 1

    if conf["CUDA_PRETEND_VERSION"]:
        EXTRA_DEFINES["CUDAPP_PRETEND_CUDA_VERSION"] = conf["CUDA_PRETEND_VERSION"]

    INCLUDE_DIRS = ['src/cpp'] + conf["BOOST_INC_DIR"]
    if conf["CUDA_INC_DIR"]:
        INCLUDE_DIRS += conf["CUDA_INC_DIR"]

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
        for lib_dir in conf["CUDADRV_LIB_DIR"]:
            conf["LDFLAGS"].extend(["-Xlinker", "-rpath", "-Xlinker", lib_dir])

    if conf["CUDA_ENABLE_GL"]:
        EXTRA_SOURCES.append("src/wrapper/wrap_cudagl.cpp")
        EXTRA_DEFINES["HAVE_GL"] = 1

    if conf["CUDA_ENABLE_CURAND"]:
        EXTRA_DEFINES["HAVE_CURAND"] = 1
        EXTRA_SOURCES.extend([
            "src/wrapper/wrap_curand.cpp"
            ])
        LIBRARIES.extend(conf["CURAND_LIBNAME"])
        LIBRARY_DIRS.extend(conf["CURAND_LIB_DIR"])

    ver_dic = {}
    exec(compile(open("pycuda/__init__.py").read(), "pycuda/__init__.py", 'exec'),
            ver_dic)

    import sys
    if sys.version_info >= (3,):
        pvt_struct_source = "src/wrapper/_pvt_struct_v3.cpp"
    else:
        pvt_struct_source = "src/wrapper/_pvt_struct_v2.cpp"

    setup(name="pycuda",
            # metadata
            version=ver_dic["VERSION_TEXT"],
            description="Python wrapper for Nvidia CUDA",
            long_description=open("README.rst", "rt").read(),
            author="Andreas Kloeckner",
            author_email="inform@tiker.net",
            license="MIT",
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
                'Programming Language :: Python :: 3',
                'Programming Language :: Python :: 2.6',
                'Programming Language :: Python :: 2.7',
                'Programming Language :: Python :: 3.3',
                'Programming Language :: Python :: 3.4',
                'Topic :: Scientific/Engineering',
                'Topic :: Scientific/Engineering :: Mathematics',
                'Topic :: Scientific/Engineering :: Physics',
                'Topic :: Scientific/Engineering :: Visualization',
                ],

            # build info
            packages=["pycuda", "pycuda.gl", "pycuda.sparse", "pycuda.compyte"],

            setup_requires=[
                "numpy>=1.6",
                ],

            install_requires=[
                "pytools>=2011.2",
                "pytest>=2",
                "decorator>=3.2.0",
                "appdirs>=1.4.0"
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
                    include_dirs=INCLUDE_DIRS,
                    library_dirs=LIBRARY_DIRS,
                    libraries=LIBRARIES,
                    define_macros=list(EXTRA_DEFINES.items()),
                    extra_compile_args=conf["CXXFLAGS"],
                    extra_link_args=conf["LDFLAGS"],
                    ),
                NumpyExtension("_pvt_struct",
                    [pvt_struct_source],
                    extra_compile_args=conf["CXXFLAGS"],
                    extra_link_args=conf["LDFLAGS"],
                    ),
                ],

            include_package_data=True,
            package_data={
                    "pycuda": [
                        "cuda/*.hpp",
                        ]
                    },

            zip_safe=False)


if __name__ == '__main__':
    main()
