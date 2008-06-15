#!/usr/bin/env python
# -*- coding: latin-1 -*-




def get_config_schema():
    from aksetup_helper import ConfigSchema, Option, \
            IncludeDir, LibraryDir, Libraries, \
            Switch, StringListOption

    return ConfigSchema([
        IncludeDir("BOOST", []),
        LibraryDir("BOOST", []),
        Libraries("BOOST_PYTHON", ["boost_python-gcc42-mt"]),

        IncludeDir("BOOST_BINDINGS", []),

        Option("CUDA_ROOT", help="Path to the CUDA toolkit"),
        Option("CUDA_BIN_DIR", help="Path to the CUDA executables"),
        IncludeDir("CUDA", None),

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
    print "*", search_path

    file_found = 0
    paths = search_path.split(pathsep)
    for path in paths:
        print path
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
            PyUblasExtension, NumpyExtension

    hack_distutils()
    conf = get_config(get_config_schema())

    LIBRARY_DIRS = conf["BOOST_LIB_DIR"]
    LIBRARIES = conf["BOOST_PYTHON_LIBNAME"]

    from os.path import dirname, join, normpath

    if conf["CUDA_ROOT"] is None:
        nvcc_path = search_on_path("nvcc")
        if nvcc_path is None:
            print "*** CUDA_ROOT not set, and nvcc not in path. Giving up."
            import sys
            sys.exit(1)
            
        conf["CUDA_ROOT"] = normpath(join(dirname(nvcc_path), ".."))

    if conf["CUDA_BIN_DIR"] is None:
        conf["CUDA_BIN_DIR"] = join(conf["CUDA_ROOT"], "bin")
    if conf["CUDA_INC_DIR"] is None:
        conf["CUDA_INC_DIR"] = [join(conf["CUDA_ROOT"], "include")]
    if conf["CUDART_LIB_DIR"] is None:
        conf["CUDART_LIB_DIR"] = [join(conf["CUDA_ROOT"], "lib")]
    if conf["CUBLAS_LIB_DIR"] is None:
        conf["CUBLAS_LIB_DIR"] = [join(conf["CUDA_ROOT"], "lib")]

    EXTRA_DEFINES = { "PYUBLAS_HAVE_BOOST_BINDINGS":1 }
    EXTRA_INCLUDE_DIRS = []
    EXTRA_LIBRARY_DIRS = []
    EXTRA_LIBRARIES = []

    INCLUDE_DIRS = [
            "src/cpp",
            ] \
            + conf["BOOST_BINDINGS_INC_DIR"] \
            + conf["BOOST_INC_DIR"] \
            + conf["CUDA_INC_DIR"] \

    conf["USE_CUDA"] = True

    setup(name="pycuda",
            # metadata
            version="0.90",
            description="Python wrapper for Nvidia CUDA",
            author=u"Andreas Kloeckner",
            author_email="inform@tiker.net",
            license = "MIT",
            url="http://mathema.tician.de/software/pycuda",
            classifiers=[
              'Environment :: Console',
              'Development Status :: 4 - Beta',
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
            packages=["pycuda"],
            zip_safe=False,

            install_requires=[
                "pytools>=3",
                ],

            package_dir={"pycuda": "src/python"},
            ext_package="pycuda",

            ext_modules=[
                NumpyExtension("_driver", 
                    [
                        "src/wrapper/wrap_cudadrv.cpp", 
                        ],
                    include_dirs=INCLUDE_DIRS + EXTRA_INCLUDE_DIRS,
                    library_dirs=LIBRARY_DIRS + conf["CUDADRV_LIB_DIR"],
                    libraries=LIBRARIES + conf["CUDADRV_LIBNAME"],
                    define_macros=list(EXTRA_DEFINES.iteritems()),
                    extra_compile_args=conf["CXXFLAGS"],
                    extra_link_args=conf["LDFLAGS"],
                    ),
            )




if __name__ == '__main__':
    main()
