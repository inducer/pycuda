# don't import pycuda.driver here--you'll create an import loop
from __future__ import annotations

import os
import sys
from os import unlink

from pytools import memoize
from pytools.prefork import call_capture_output
from pycuda.compiler import CudaModule, _find_pycuda_include_path, _new_md5
import glob

@memoize
def get_nvfortran_version(nvfortran):
    cmdline = [nvfortran, "--version"]
    try:
        result, stdout, _stderr = call_capture_output(cmdline)
    except Exception:
        raise SystemExit("nvfortran not found")

    if result != 0 or not stdout:
        from warnings import warn

        warn("nvfortran version could not be determined.", stacklevel=2)
        stdout = b"nvfortran unknown version"

    return stdout.decode("utf-8", "replace")

def compile_plain_fortran(source, options, keep, nvfortran, cache_dir, target="cubin"):
    from os.path import join

    assert target in ["cubin"]

    if cache_dir:
        checksum = _new_md5()
        checksum.update(source.encode("utf-8"))

        for option in options:
            checksum.update(option.encode("utf-8"))
        checksum.update(get_nvfortran_version(nvfortran).encode("utf-8"))
        from pycuda.characterize import platform_bits

        checksum.update(str(platform_bits()).encode("utf-8"))

        cache_file = checksum.hexdigest()
        cache_path = join(cache_dir, cache_file + "." + target)

        try:
            with open(cache_path, "rb") as cache_file:
                return cache_file.read()

        except Exception:
            pass

    from tempfile import mkdtemp

    file_dir = mkdtemp()
    file_root = "kernel"
    cu_file_name = file_root + ".cuf"
    cu_file_path = join(file_dir, cu_file_name)

    with open(cu_file_path, "w") as outf:
        outf.write(str(source))

    cmdline = [nvfortran, "-gpu:keep", *options, cu_file_name]

    result, stdout, stderr = call_capture_output(
        cmdline, cwd=file_dir, error_on_nonzero=False
    )

    try:
        fwild = glob.glob(join(file_dir,"pgcuda.*." + target))[0]
        result_f = open(fwild,"rb")
    except OSError:
        no_output = True
    else:
        no_output = False

    if result != 0 or (no_output and (stdout or stderr)):
        if result == 0:
            from warnings import warn

            warn(
                "PyCUDA: nvfortran exited with status 0, but appears to have "
                "encountered an error", stacklevel=2
            )
        from pycuda.driver import CompileError

        raise CompileError(
            "nvfortran compilation of %s failed" % cu_file_path,
            cmdline,
            stdout=stdout.decode("utf-8", "replace"),
            stderr=stderr.decode("utf-8", "replace"),
        )

    if stdout or stderr:
        lcase_err_text = (stdout + stderr).decode("utf-8", "replace").lower()
        from warnings import warn

        if "demoted" in lcase_err_text or "demoting" in lcase_err_text:
            warn(
                "nvfortran said it demoted types in source code it "
                "compiled--this is likely not what you want.",
                stacklevel=4,
            )
        warn(
            "The CUDA compiler succeeded, but said the following:\n"
            + (stdout + stderr).decode("utf-8", "replace"),
            stacklevel=4,
        )

    result_data = result_f.read()
    result_f.close()

    if cache_dir:
        with open(cache_path, "wb") as outf:
            outf.write(result_data)

    if not keep:
        from os import listdir, rmdir, unlink

        for name in listdir(file_dir):
            unlink(join(file_dir, name))
        rmdir(file_dir)

    return result_data

DEFAULT_NVFORTRAN_FLAGS = [
    _flag.strip()
    for _flag in os.environ.get("PYCUDA_DEFAULT_NVFORTRAN_FLAGS", "").split()
    if _flag.strip()
]

def compilefortran(
    source,
    nvfortran="nvfortran",
    options=None,
    keep=False,
    no_main_f=False,
    arch=None,
    cache_dir=None,
    include_dirs=None,
    target="cubin",
):

    if include_dirs is None:
        include_dirs = []
    assert target in ["cubin"]

    if not no_main_f:
        source = '%s\n end\n' % source

    if options is None:
        options = DEFAULT_NVFORTRAN_FLAGS

    options = options[:]
    if arch is None:
        from pycuda.driver import Error

        try:
            from pycuda.driver import Context

            arch = "sm_%d%d" % Context.get_device().compute_capability()
        except Error:
            pass

    from pycuda.driver import CUDA_DEBUGGING

    if CUDA_DEBUGGING:
        cache_dir = False
        keep = True
        options.extend(["-g", "-G"])

    if "PYCUDA_CACHE_DIR" in os.environ and cache_dir is None:
        cache_dir = os.environ["PYCUDA_CACHE_DIR"]

    if "PYCUDA_DISABLE_CACHE" in os.environ:
        cache_dir = False

    if cache_dir is None:
        import platformdirs

        cache_dir = os.path.join(
            platformdirs.user_cache_dir("pycuda", "pycuda"), "compiler-cache-v1"
        )

        from os import makedirs
        makedirs(cache_dir, exist_ok=True)

    if arch is not None:
        options.extend(["-gpu:"+arch])

    if (
            ("darwin" in sys.platform and sys.maxsize == 9223372036854775807)
            or
            ("win32" in sys.platform and sys.maxsize == 9223372036854775807)):
        options.append("-m64")
    elif "win32" in sys.platform and sys.maxsize == 2147483647:
        options.append("-m32")

    include_dirs = [*include_dirs, _find_pycuda_include_path()]

    for i in include_dirs:
        options.append("-I" + i)

    return compile_plain_fortran(source, options, keep, nvfortran, cache_dir, target)

class SourceModuleFortran(CudaModule):
    """
    Creates a Module from a single .cuf source object linked against the
    static CUDA runtime.
    """

    def __init__(
        self,
        source,
        nvfortran="nvfortran",
        options=None,
        keep=False,
        no_main_f=False,
        arch=None,
        cache_dir=None,
        include_dirs=None,
    ):
        if include_dirs is None:
            include_dirs = []
        self._check_arch(arch)

        cubin = compilefortran(
            source,
            nvfortran,
            options,
            keep,
            no_main_f,
            arch,
            cache_dir,
            include_dirs
        )

        from pycuda.driver import module_from_buffer

        self.module = module_from_buffer(cubin)

        self._bind_module()
