from __future__ import absolute_import
from __future__ import print_function
from pytools import memoize
# don't import pycuda.driver here--you'll create an import loop
import sys
from tempfile import mkstemp
from os import unlink

from pytools.prefork import call_capture_output
from six.moves import map


@memoize
def get_nvcc_version(nvcc):
    cmdline = [nvcc, "--version"]
    result, stdout, stderr = call_capture_output(cmdline)

    if result != 0 or not stdout:
        from warnings import warn
        warn("NVCC version could not be determined.")
        stdout = "nvcc unknown version"

    return stdout.decode("utf-8", "replace")


def _new_md5():
    try:
        import hashlib
        return hashlib.md5()
    except ImportError:
        # for Python << 2.5
        import md5
        return md5.new()


def preprocess_source(source, options, nvcc):
    handle, source_path = mkstemp(suffix='.cu')

    outf = open(source_path, 'w')
    outf.write(source)
    outf.close()
    os.close(handle)

    cmdline = [nvcc, '--preprocess'] + options + [source_path]
    if 'win32' in sys.platform:
        cmdline.extend(['--compiler-options', '-EP'])
    else:
        cmdline.extend(['--compiler-options', '-P'])

    result, stdout, stderr = call_capture_output(cmdline, error_on_nonzero=False)

    if result != 0:
        from pycuda.driver import CompileError
        raise CompileError("nvcc preprocessing of %s failed" % source_path,
                           cmdline, stderr=stderr)

    # sanity check
    if len(stdout) < 0.5*len(source):
        from pycuda.driver import CompileError
        raise CompileError("nvcc preprocessing of %s failed with ridiculously "
                "small code output - likely unsupported compiler." % source_path,
                cmdline, stderr=stderr.decode("utf-8", "replace"))

    unlink(source_path)

    return stdout.decode("utf-8", "replace")


def compile_plain(source, options, keep, nvcc, cache_dir, target="cubin"):
    from os.path import join

    assert target in ["cubin", "ptx", "fatbin"]

    if cache_dir:
        checksum = _new_md5()

        if '#include' in source:
            checksum.update(preprocess_source(source, options, nvcc).encode("utf-8"))
        else:
            checksum.update(source.encode("utf-8"))

        for option in options:
            checksum.update(option.encode("utf-8"))
        checksum.update(get_nvcc_version(nvcc).encode("utf-8"))
        from pycuda.characterize import platform_bits
        checksum.update(str(platform_bits()).encode("utf-8"))

        cache_file = checksum.hexdigest()
        cache_path = join(cache_dir, cache_file + "." + target)

        try:
            cache_file = open(cache_path, "rb")
            try:
                return cache_file.read()
            finally:
                cache_file.close()

        except:
            pass

    from tempfile import mkdtemp
    file_dir = mkdtemp()
    file_root = "kernel"

    cu_file_name = file_root + ".cu"
    cu_file_path = join(file_dir, cu_file_name)

    outf = open(cu_file_path, "w")
    outf.write(str(source))
    outf.close()

    if keep:
        options = options[:]
        options.append("--keep")

        print("*** compiler output in %s" % file_dir)

    cmdline = [nvcc, "--" + target] + options + [cu_file_name]
    result, stdout, stderr = call_capture_output(cmdline,
            cwd=file_dir, error_on_nonzero=False)

    try:
        result_f = open(join(file_dir, file_root + "." + target), "rb")
    except IOError:
        no_output = True
    else:
        no_output = False

    if result != 0 or (no_output and (stdout or stderr)):
        if result == 0:
            from warnings import warn
            warn("PyCUDA: nvcc exited with status 0, but appears to have "
                    "encountered an error")
        from pycuda.driver import CompileError
        raise CompileError("nvcc compilation of %s failed" % cu_file_path,
                cmdline, stdout=stdout.decode("utf-8", "replace"),
                stderr=stderr.decode("utf-8", "replace"))

    if stdout or stderr:
        lcase_err_text = (stdout+stderr).decode("utf-8", "replace").lower()
        from warnings import warn
        if "demoted" in lcase_err_text or "demoting" in lcase_err_text:
            warn("nvcc said it demoted types in source code it "
                "compiled--this is likely not what you want.",
                stacklevel=4)
        warn("The CUDA compiler succeeded, but said the following:\n"
                + (stdout+stderr).decode("utf-8", "replace"), stacklevel=4)

    result_data = result_f.read()
    result_f.close()

    if cache_dir:
        outf = open(cache_path, "wb")
        outf.write(result_data)
        outf.close()

    if not keep:
        from os import listdir, unlink, rmdir
        for name in listdir(file_dir):
            unlink(join(file_dir, name))
        rmdir(file_dir)

    return result_data


def _get_per_user_string():
    try:
        from os import getuid
    except ImportError:
        checksum = _new_md5()
        from os import environ
        checksum.update(environ["USERNAME"].encode("utf-8"))
        return checksum.hexdigest()
    else:
        return "uid%d" % getuid()


def _find_pycuda_include_path():
    from pkg_resources import Requirement, resource_filename
    return resource_filename(Requirement.parse("pycuda"), "pycuda/cuda")


import os
DEFAULT_NVCC_FLAGS = [
        _flag.strip() for _flag in
        os.environ.get("PYCUDA_DEFAULT_NVCC_FLAGS", "").split()
        if _flag.strip()]


def compile(source, nvcc="nvcc", options=None, keep=False,
        no_extern_c=False, arch=None, code=None, cache_dir=None,
        include_dirs=[], target="cubin"):

    assert target in ["cubin", "ptx", "fatbin"]

    if not no_extern_c:
        source = 'extern "C" {\n%s\n}\n' % source

    if options is None:
        options = DEFAULT_NVCC_FLAGS

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
        from os.path import join
        import appdirs
        cache_dir = os.path.join(appdirs.user_cache_dir("pycuda", "pycuda"),
                "compiler-cache-v1")

        from os import makedirs
        try:
            makedirs(cache_dir)
        except OSError as e:
            from errno import EEXIST
            if e.errno != EEXIST:
                raise

    if arch is not None:
        options.extend(["-arch", arch])

    if code is not None:
        options.extend(["-code", code])

    if 'darwin' in sys.platform and sys.maxsize == 9223372036854775807:
        options.append('-m64')
    elif 'win32' in sys.platform and sys.maxsize == 9223372036854775807:
        options.append('-m64')
    elif 'win32' in sys.platform and sys.maxsize == 2147483647:
        options.append('-m32')

    include_dirs = include_dirs + [_find_pycuda_include_path()]

    for i in include_dirs:
        options.append("-I"+i)

    return compile_plain(source, options, keep, nvcc, cache_dir, target)

class JitLinkModule(object):
    def __init__(self, nvcc='nvcc', options=None, keep=False,
            no_extern_c=False, arch=None, code=None, cache_dir=None,
            include_dirs=[],  message_handler=None, log_verbose=False,
            cudalib_dir=None):
        self._check_arch(arch)
        self.nvcc = nvcc
        self.keep = keep
        self.no_extern_c = no_extern_c
        self.arch = arch
        self.code = code
        self.cache_dir = cache_dir
        self.include_dirs = include_dirs
        self.cudalib_dir = cudalib_dir
        self.module = None
        from pycuda.driver import Linker
        self.linker = Linker(message_handler, options, log_verbose)

    def add_source(self, source, nvcc_options=None, name='kernel.ptx'):
        ptx = compile(source, nvcc=self.nvcc, options=nvcc_options,
            keep=self.keep, no_extern_c=self.no_extern_c, arch=self.arch,
            code=self.code, cache_dir=self.cache_dir,
            include_dirs=self.include_dirs, target="ptx")
        from pycuda.driver import jit_input_type
        self.linker.add_data(ptx, jit_input_type.PTX, name)

    def add_data(self, data, input_type, name='unknown'):
        self.linker.add_data(data, input_type, name)

    def add_file(self, filename, input_type):
        self.linker.add_file(filename, input_type)

    def add_stdlib(self, libname):
        import os, os.path
        from pycuda.driver import jit_input_type
        from platform import system
        syst = system()
        if syst == 'Windows':
            if self.cudalib_dir is None:
                libdir = os.path.join(os.environ['CUDA_PATH'], 'lib/x64')
            else:
                libdir = self.cudalib_dir
            libpath = os.path.join(libdir, '%s.lib' % libname)
        elif syst == 'Linux':
            if self.cudalib_dir is None:
                libdir = '/usr/lib/x86_64-linux-gnu'
            else:
                libdir = self.cudalib_dir
            libpath = os.path.join(libdir, 'lib%s.a' % libname)
        else:
            raise Exception('System "%s" currently not supported, use add_file() instead' % syst)
        if os.path.isfile(libpath):
            self.linker.add_file(libpath, jit_input_type.LIBRARY)
        else:
            raise Exception('Library file "%s" not found, use add_file() instead' % libpath)

    def link(self):
        self.module = self.linker.link_module()
        self.linker = None
        self.get_global = self.module.get_global
        self.get_texref = self.module.get_texref
        if hasattr(self.module, "get_surfref"):
            self.get_surfref = self.module.get_surfref

    def _check_arch(self, arch):
        if arch is None:
            return
        try:
            from pycuda.driver import Context
            capability = Context.get_device().compute_capability()
            if tuple(map(int, tuple(arch.split("_")[1]))) > capability:
                from warnings import warn
                warn("trying to compile for a compute capability "
                        "higher than selected GPU")
        except:
            pass

    def get_function(self, name):
        return self.module.get_function(name)

class SourceModule(JitLinkModule):
    def __init__(self, source, nvcc="nvcc", options=None, keep=False,
            no_extern_c=False, arch=None, code=None, cache_dir=None,
            include_dirs=[], message_handler=None, log_verbose=False):
        super(SourceModule, self).__init__(nvcc=nvcc, options=None,
            keep=keep, no_extern_c=no_extern_c, arch=arch, code=code,
            cache_dir=cache_dir, include_dirs=include_dirs,
            message_handler=message_handler, log_verbose=log_verbose)
        self.add_source(source, nvcc_options=options)
        self.link()
