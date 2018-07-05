"""
This exports a "deferred" implementation of SourceModule, where compilation
is delayed until call-time.  Several methods, like get_function(), return
"deferred" values that are also only evaluated at call-time.
"""

from pycuda.tools import context_dependent_memoize
from pycuda.compiler import compile, SourceModule
import pycuda.driver

import re

class DeferredSource(object):
    '''
    Source generator that supports user-directed indentation, nesting
    ``DeferredSource`` objects, indentation-aware string interpolation,
    and deferred generation.
    Use ``+=`` or ``add()`` to add source fragments as strings or
    other ``DeferredSource`` objects, ``indent()`` or ``dedent()`` to
    change base indentation, and ``__call__`` or ``generate()`` to
    generate source.
    
    '''
    def __init__(self, str=None, subsources=None, base_indent=0, indent_step=2):
        self.base_indent = base_indent
        self.indent_step = indent_step
        if subsources is None:
            subsources = []
        self.subsources = subsources
        if str is not None:
            self.add(str)
        self._regex_space = re.compile(r"^(\s*)(.*?)(\s*)$")
        self._regex_format = re.compile(r"%\(([^\)]*)\)([a-zA-Z])")

    def __str__(self):
        return self.generate()

    def __repr__(self):
        return repr(self.__str__())

    def __call__(self, indent=0, indent_first=True):
        return self.generate(indent, indent_first)

    def generate(self, indent=0, indent_first=True, get_list=False):
        if get_list:
            retval = []
        else:
            retval = ''
        do_indent = not indent_first
        for subindent, strip_space, subsource, format_dict in self.subsources:
            if do_indent:
                newindent = self.base_indent + indent + subindent
            else:
                newindent = 0
            do_indent = True
            if isinstance(subsource, DeferredSource):
                retval = retval + subsource.generate(indent=(indent + subindent), get_list=get_list)
                continue
            lines = subsource.split("\n")
            minstrip = None
            newlines = []
            for line in lines:
                linelen = len(line)
                space_match = self._regex_space.match(line)
                end_leading_space = space_match.end(1)
                begin_trailing_space = space_match.start(3)
                if strip_space:
                    if linelen == end_leading_space:
                        # all space, ignore
                        continue
                    if minstrip is None or end_leading_space < minstrip:
                        minstrip = end_leading_space
                if not format_dict:
                    newlines.append(line)
                    continue
                newlinelist = None
                newline = ''
                curpos = 0
                matches = list(self._regex_format.finditer(line, end_leading_space))
                nummatches = len(matches)
                for match in matches:
                    formatchar = match.group(2)
                    name = match.group(1)
                    matchstart = match.start()
                    matchend = match.end()
                    repl = format_dict.get(name, None)
                    if repl is None:
                        continue
                    if (isinstance(repl, DeferredSource) and
                        nummatches == 1 and
                        matchstart == end_leading_space):
                        # only one replacement, and only spaces preceding
                        space = space_match.group(1)
                        newlinelist = [ space + x
                                        for x in repl.generate(get_list=True) ]
                    else:
                        newline = newline + line[curpos:matchstart]
                        newline = newline + (('%' + formatchar) % (repl,))
                    curpos = matchend
                if newlinelist:
                    newlines.extend(newlinelist)
                else:
                    newlines.append(newline)
                # add remaining unprocessed part of line to end of last
                # replacement
                newlines[-1] = newlines[-1] + line[curpos:]
            indentstr = ' ' * (indent + subindent)
            for i, line in enumerate(newlines):
                line = indentstr + line[minstrip:]
                newlines[i] = line
            if get_list:
                retval += newlines
            else:
                retval = retval + "\n".join(newlines) + "\n"
        return retval

    def indent(self, indent_step=None):
        if indent_step is None:
            indent_step = self.indent_step
        self.base_indent += indent_step
        return self

    def dedent(self, indent_step=None):
        if indent_step is None:
            indent_step = self.indent_step
        self.base_indent -= indent_step
        return self

    def format_dict(self, format_dict):
        for subsource in self.subsources:
            subsource[3] = format_dict
        return self

    def add(self, other, strip_space=True, format_dict=None):
        self.subsources.append([self.base_indent, strip_space, other, format_dict])
        return self

    def __iadd__(self, other):
        self.add(other)
        return self

    def __add__(self, other):
        newgen = DeferredSource(subsources=self.subsources,
                                base_indent=self.base_indent,
                                indent_step=self.indent_step)
        newgen.add(other)
        return newgen

class DeferredVal(object):
    '''
    This is an object that serves as a proxy to an as-yet undetermined
    object, which is only known at the time when either ``_set_val()``
    or ``_evalbase()`` is called.  Any calls to methods listed in the class
    attribute ``_deferred_method_dict`` are queued until a call to
    ``_eval()``, at which point ``_evalbase()`` is called (unless the value
    has been already set by an earlier call to ``_evalbase()`` or
    ``set_val()``), and then the queued method calls are executed, in order,
    immediately on the new object.
    This class must be subclassed, and the class attribute
    ``_deferred_method_dict`` must contain a mapping from defer-able method
    names to either ``DeferredVal``, None (same as ``DeferredVal``), or a
    subclass, which when instantiated, will be assigned (with ``_set_val()``)
    the return value of the method.
    There are two ways to set the proxied object.  One is to set it
    explicitly with ``_set_val(val)``.  The other is to override the method
    ``_evalbase()`` which should return the new object, and will be called
    by ``_eval()``.
    '''
    __unimpl = object()
    _deferred_method_dict = None # must be set by subclass

    def __init__(self):
        self._val_available = False
        self._val = None
        self._deferred_method_calls = []

    def _copy(self, retval=None):
        if retval is None:
            retval = self.__class__()
        retval._val_available = self._val_available
        retval._val = self._val
        retval._deferred_method_calls = self._deferred_method_calls
        return retval

    def __repr__(self):
        return self._repr(0)

    def _repr(self, indent):
        indentstr = " " * indent
        retstrs = []
        retstrs.append("")
        retstrs.append("%s [" % (self.__class__,))
        for dmc in self._deferred_method_calls:
            (name, args, kwargs, retval) = dmc
            retstrs.append("  method %s" % (repr(name),))
            for arg in args:
                if isinstance(arg, DeferredVal):
                    retstrs.append("    deferred arg (id=%s)" % (id(arg),))
                    retstrs.append(arg._repr(indent + 6))
                else:
                    retstrs.append("    arg %s" % (repr(arg),))
            for kwname, arg in kwargs.items():
                if isinstance(arg, DeferredVal):
                    retstrs.append("    deferred kwarg %s (id=%s)" % (repr(kwname), id(arg)))
                    retstrs.append(arg._repr(indent + 6))
                else:
                    retstrs.append("    kwarg %s=%s" % (kwname, repr(arg),))
            retstrs.append("    deferred retval (id=%s)" % (id(retval),))
        retstrs.append("]")
        return "\n".join([(indentstr + retstr) for retstr in retstrs])

    def _set_val(self, val):
        self._val = val
        self._val_available = True
        self._eval_methods()
        return val

    def _evalbase(self, evalcont=None):
        raise NotImplementedError()

    def _eval_list(self, vals, evalcont=None):
        newvals = []
        for val in vals:
            if isinstance(val, DeferredVal):
                val = val._eval(evalcont=evalcont)
            newvals.append(val)
        return newvals

    def _eval_dict(self, valsdict, evalcont=None):
        newvalsdict = {}
        for name, val in valsdict.items():
            if isinstance(val, DeferredVal):
                val = val._eval(evalcont=evalcont)
            newvalsdict[name] = val
        return newvalsdict

    def _eval_methods(self, evalcont=None):
        assert(self._val_available)
        val = self._val
        for op in self._deferred_method_calls:
            (methodname, methodargs, methodkwargs, deferredretval) = op
            methodargs = self._eval_list(methodargs, evalcont=evalcont)
            methodkwargs = self._eval_dict(methodkwargs, evalcont=evalcont)
            retval = getattr(val, methodname)(*methodargs, **methodkwargs)
            deferredretval._set_val(retval)
        self._deferred_method_calls = []

    def _eval(self, evalcont=None):
        if not self._val_available:
            self._val = self._evalbase(evalcont=evalcont)
            self._val_available = True
            self._eval_methods(evalcont=evalcont)
        return self._val

    def _get_deferred_func(self, _name, _retval):
        def _deferred_func(*args, **kwargs):
            if not self._val_available:
                self._deferred_method_calls.append((_name, args, kwargs, _retval))
                return _retval
            args = self._eval_list(args)
            kwargs = self._eval_dict(kwargs)
            return getattr(self._val, _name)(*args, **kwargs)
        _deferred_func.__name__ = _name + ".deferred"
        return _deferred_func

    def __getattr__(self, name):
        if self.__class__._deferred_method_dict is None:
            raise Exception("DeferredVal must be subclassed and the class attribute _deferred_method_dict must be set to a valid dictionary!")
        if self._val_available:
            return getattr(self._val, name)
        deferredclass = self.__class__._deferred_method_dict.get(name, self.__unimpl)
        if deferredclass is not self.__unimpl:
            if deferredclass is None:
                deferredclass = DeferredVal
            retval = deferredclass()
            return self._get_deferred_func(name, retval)
        raise AttributeError("no such attribute (yet): '%s'" % (name,))

# we allow all math operators to be deferred
_mathops = (
    '__add__', '__sub__', '__mul__', '__floordiv__', '__mod__',
    '__divmod__', '__pow__', '__lshift__', '__rshift__', '__and__',
    '__xor__', '__or__', '__div__', '__truediv__', '__radd__', '__rsub__',
    '__rmul__', '__rdiv__', '__rtruediv__', '__rfloordiv__', '__rmod__',
    '__rdivmod__', '__rpow__', '__rlshift__', '__rrshift__', '__rand__',
    '__rxor__', '__ror__', '__iadd__', '__isub__', '__imul__', '__idiv__',
    '__itruediv__', '__ifloordiv__', '__imod__', '__ipow__', '__ilshift__',
    '__irshift__', '__iand__', '__ixor__', '__ior__', '__pos__', '__abs__',
    '__invert__', '__complex__', '__int__', '__long__', '__float__',
    '__oct__', '__hex__', '__index__', '__coerce__')
class DeferredNumeric(DeferredVal):
    pass
DeferredNumeric._deferred_method_dict = dict((x, DeferredNumeric)
                                             for x in _mathops)
def _get_deferred_attr_func(_name):
    def _deferred_func(self, *args, **kwargs):
        return self.__getattr__(_name)(*args, **kwargs)
    _deferred_func.__name__ = _name
    return _deferred_func
for name in _mathops:
    setattr(DeferredNumeric, name, _get_deferred_attr_func(name))

class DeferredModuleCall(DeferredVal):
    _deferred_method_dict = {}
    def __init__(self, methodstr, *args, **kwargs):
        super(DeferredModuleCall, self).__init__()
        self._methodstr = methodstr
        self._args = args
        self._kwargs = kwargs

    def _copy(self, retval=None):
        if retval is None:
            retval = self.__class__(self._methodstr, *self._args, **self._kwargs)
        return super(DeferredModuleCall, self)._copy(retval=retval)

    def _evalbase(self, evalcont):
        # evalcont is the actual Module object
        return getattr(evalcont, self._methodstr)(*self._args, **self._kwargs)

class DeferredTexRef(DeferredModuleCall):
    _deferred_method_dict = {
        "set_array": None,
        "set_address": DeferredNumeric,
        "set_address_2d": None,
        "set_format": None,
        "set_address_mode": None,
        "set_flags": None,
        "get_address": DeferredNumeric,
        "get_flags": DeferredNumeric,
    }

class DeferredFunction(object):
    '''
    This class is a pseudo-replacement of ``pycuda.driver.Function``,
    but takes a ``DeferredSourceModule`` and a function name as an argument,
    and queues any call to ``prepare()`` until call-time, at which time it
    calls out to the ``DeferredSourceModule`` object do create the actual
    Function before preparing (if necessary) and calling the underlying
    kernel.  NOTE: you may now send the actual ``GPUArrays`` as arguments,
    rather than their ``.gpudata`` members; this can be helpful to
    dynamically create kernels.
    '''
    def __init__(self, deferredmod, funcname):
        self._deferredmod = deferredmod
        self._funcname = funcname
        self._prepare_args = None

        def get_unimplemented(_methodname):
            def _unimplemented(self, _methodname=_methodname, *args, **kwargs):
                raise NotImplementedError("%s does not implement method '%s'" % (type(self), _methodname,))
            return _unimplemented

        for meth_name in ["set_block_shape", "set_shared_size",
                "param_set_size", "param_set", "param_seti", "param_setf",
                "param_setv", "param_set_texref",
                "launch", "launch_grid", "launch_grid_async"]:
            setattr(self, meth_name, get_unimplemented(meth_name))

    def _fix_texrefs(self, kwargs, mod):
        texrefs = kwargs.get('texrefs', None)
        if texrefs is not None:
            kwargs = kwargs.copy()
            newtexrefs = []
            for texref in texrefs:
                if isinstance(texref, DeferredVal):
                    texref = texref._copy() # future calls may use different modules/functions
                    texref = texref._eval(mod)
                newtexrefs.append(texref)
            kwargs['texrefs'] = newtexrefs
        return kwargs

    def __call__(self, *args, **kwargs):
        block = kwargs.get('block', None)
        if block is None or not isinstance(block, tuple) or len(block) != 3:
            raise ValueError("keyword argument 'block' is required, and must be a 3-tuple of integers")
        grid = kwargs.get('grid', (1,1))
        mod, func = self._deferredmod._delayed_get_function(self._funcname, args, grid, block)
        kwargs = self._fix_texrefs(kwargs, mod)
        return func.__call__(*args, **kwargs)

    def param_set_texref(self, *args, **kwargs):
        raise NotImplementedError()

    def prepare(self, *args, **kwargs):
        self._prepare_args = (args, kwargs)
        return self

    def _do_delayed_prepare(self, mod, func):
        if self._prepare_args is None:
            raise Exception("prepared_*_call() requires that prepare() be called first")
        (prepare_args, prepare_kwargs) = self._prepare_args
        prepare_kwargs = self._fix_texrefs(prepare_kwargs, mod)
        func.prepare(*prepare_args, **prepare_kwargs)

    def _generic_prepared_call(self, funcmethodstr, funcmethodargs, funcargs, funckwargs):
        grid = funcmethodargs[0]
        block = funcmethodargs[1]
        mod, func = self._deferredmod._delayed_get_function(self._funcname, funcargs, grid, block)
        self._do_delayed_prepare(mod, func)
        newfuncargs = [ getattr(arg, 'gpudata', arg) for arg in funcargs ]
        fullargs = list(funcmethodargs)
        fullargs.extend(newfuncargs)
        return getattr(func, funcmethodstr)(*fullargs, **funckwargs)

    def prepared_call(self, grid, block, *args, **kwargs):
        return self._generic_prepared_call('prepared_call', (grid, block), args, kwargs)

    def prepared_timed_call(self, grid, block, *args, **kwargs):
        return self._generic_prepared_call('prepared_timed_call', (grid, block), args, kwargs)

    def prepared_async_call(self, grid, block, stream, *args, **kwargs):
        return self._generic_prepared_call('prepared_async_call', (grid, block, stream), args, kwargs)

@context_dependent_memoize
def _delayed_compile_aux(source, compileargs):
    # re-convert any tuples to lists
    newcompileargs = []
    for i, arg in enumerate(compileargs):
        if isinstance(arg, tuple):
            arg = list(arg)
        newcompileargs.append(arg)
    cubin = compile(source, *newcompileargs)

    from pycuda.driver import module_from_buffer
    return module_from_buffer(cubin)

class DeferredSourceModule(SourceModule):
    '''
    This is an abstract specialization of SourceModule which allows the
    delay of creating the actual kernel source until call-time, at which
    point the actual arguments are available and their characteristics can
    be used to create specific kernels.
    To support this, ``get_function()`` returns a ``DeferredFunction``
    object which queues any calls to ``DeferredFunction.prepare()`` and
    saves them until future calls to ``DeferredFunction.__call__()`` or
    ``DeferredFunction.prepared_*_call()``.  NOTE: you may now send actual
    ``GPUArrays`` to these functions rather their ``.gpudata`` members;
    this can be helpful when creating dynamic kernels.
    Likewise, ``get_global()``, ``get_texref()`` and ``get_surfref()``
    return proxy objects that can be stored by ``DeferredFunction.prepare()``
    and will only be evaluated at call-time.
    This class must be subclassed and the function ``create_source(self,
    grid, block, *args)`` must be overridden, returning the kernel source
    (or ``DeferredSource`` object) that should be compiled.  ``grid``,
    ``block``, and ``*args`` are the same arguments that were sent to the
    ``DeferredFunction`` call functions above.
    The function ``create_key(self, grid, block, *args)`` returns a tuple
    ``(key, precalc)``.  ``create_key`` is always called before
    ``create_source`` and any pre-calculated info in ``precalc`` is sent back
    to ``create_source``; ``key`` is used to cache any compiled functions.
    Default return value of ``create_key()`` is (None, None), which means to
    use the function name and generated source as the key.  The key returned
    by ``create_key()`` must be usable as a hash key.
    '''
    _cache = {}

    def __init__(self, nvcc="nvcc", options=None, keep=False,
            no_extern_c=False, arch=None, code=None, cache_dir=None,
            include_dirs=[]):
        self._arch = arch
        # tuples below are so _compileargs can be used as a hash key
        if options is not None:
            options = tuple(options)
        include_dirs = tuple(include_dirs)
        self._compileargs = (nvcc, options, keep, no_extern_c,
                             arch, code, cache_dir, include_dirs)

    def _delayed_compile(self, source):
        self._check_arch(self._arch)

        return _delayed_compile_aux(source, self._compileargs)

    def create_key(self, grid, block, *funcargs):
        return (None, None)

    def create_source(self, precalc, grid, block, *funcargs):
        raise NotImplementedError("create_source must be overridden!")

    def _delayed_get_function(self, funcname, funcargs, grid, block):
        '''
        If the first element of the tuple returned by ``create_key()`` is
        not None, then it is used as the key to cache compiled functions.
        Otherwise the return value of ``create_source()`` is used as the key.
        '''
        context = pycuda.driver.Context.get_current()
        funccache = DeferredSourceModule._cache.get(context, None)
        if funccache is None:
            funccache = self._cache[context] = {}
        (key, precalc) = self.create_key(grid, block, *funcargs)
        funckey = (funcname, key)
        if key is None or funckey not in funccache:
            source = self.create_source(precalc, grid, block, *funcargs)
            if isinstance(source, DeferredSource):
                source = source.generate()
            if key is None:
                funckey = (funcname, source)
        modfunc = funccache.get(funckey, None)
        if modfunc is None:
            module = self._delayed_compile(source)
            func = module.get_function(funcname)
            modfunc = funccache[funckey] = (module, func)
        return modfunc

    def get_function(self, name):
        return DeferredFunction(self, name)

    def get_global(self, name):
        raise NotImplementedError("Deferred globals in element-wise kernels not supported yet")

    def get_texref(self, name):
        return DeferredTexRef('get_texref', name)

    def get_surfref(self, name):
        raise NotImplementedError("Deferred surfaces in element-wise kernels not supported yet")

