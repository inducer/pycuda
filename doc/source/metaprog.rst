.. _metaprog:

Metaprogramming
===============

In 'conventional' programming, one writes a program that accomplishes a
task. In *metaprogramming*, one writes a program *that writes a program*
that accomplishes a task.

That sounds pretty complicated--so first of all, we'll look at why it may
be a good idea nonetheless.

Why Metaprogramming?
--------------------

Automated Tuning
^^^^^^^^^^^^^^^^

A sizable part of a CUDA programmer's time is typically spent tuning code.
This tuning answers questions like:

 * What's the optimal number of threads per block?
 * How much data should I work on at once?
 * What data should be loaded into shared memory, and how big should the
   corresponding blocks be?

If you are lucky, you'll be able to find a pattern in the execution
time of your code and come up with a heuristic that will allow you to
reliably pick the fastest version. Unfortunately, this heuristic may
become unreliable or even fail entirely with new hardware generations.
The solution to this problem that PyCUDA tries to promote is:

   Forget heuristics. Benchmark at run time and use whatever works fastest.

This is an important advantage of PyCUDA over the CUDA runtime API: It lets
you make these decisions *while your code is running*. A number of prominent
computing packages make use of a similar technique, among them ATLAS and 
FFTW. And while those require rather complicated optimization driver 
routines, you can drive PyCUDA from the comfort of Python.

Data Types
^^^^^^^^^^

Your code may have to deal with different data types at run time. It may,
for example, have to work on both single and double precision floating
point numbers. You could just precompile versions for both, but why?
Just generate whatever code is needed right *when* it is needed.

Specialize Code for the Given Problem
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If you are writing a library, then your users will ask your library 
to perform a number of tasks. Imagine how liberating it would be if you
could generate code purposely for the problem you're being asked to 
solve, instead of having to keep code unnecessarily generic and thereby
slow. PyCUDA makes this a reality.

Constants are Faster than Variables
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If your problem sizes vary from run to run, but you perform a larger
number of kernel invocations on data of identical size, you may want
to consider compiling data size into your code as a constant. This can
have significant performance benefits, resulting mainly from decreased
fetch times and less register pressure. In particular, multiplications 
by constants are much more efficiently carried out than general
variable-variable multiplications.

Loop Unrolling
^^^^^^^^^^^^^^

The CUDA programming guide says great things about :command:`nvcc` and how
it will unroll loops for you. As of Version 2.1, that's simply not true, and
``#pragma unroll`` is simply a no-op, at least according to my experience.
With metaprogramming, you can dynamically unroll your loops to the needed
size in Python.

Metaprogramming using a Templating Engine
-----------------------------------------

If your metaprogramming needs are rather simple, perhaps the easiest way
to generate code at run time is through a templating engine. Many 
templating engines for Python exist, two of the most prominent ones are
`Jinja 2 <http://jinja.pocoo.org/>`_ and
`Cheetah <http://www.cheetahtemplate.org/>`_.

The following is a simple metaprogram that performs vector addition on
configurable block sizes. It illustrates the templating-based 
metaprogramming technique::

    from jinja2 import Template

    tpl = Template("""
        __global__ void add(
                {{ type_name }} *tgt, 
                {{ type_name }} *op1, 
                {{ type_name }} *op2)
        {
          int idx = threadIdx.x + 
            {{ thread_block_size }} * {{block_size}}
            * blockIdx.x;

          {% for i in range(block_size) %}
              {% set offset = i*thread_block_size %}
              tgt[idx + {{ offset }}] = 
                op1[idx + {{ offset }}] 
                + op2[idx + {{ offset }}];
          {% endfor %}
        }""")

    rendered_tpl = tpl.render(
        type_name="float", block_size=block_size,
        thread_block_size=thread_block_size)

    mod = SourceModule(rendered_tpl)

This snippet in a working context can be found in 
:file:`examples/demo_meta_template.py`.

You can also find an example of matrix multiplication optimization
using template metaprogramming with Cheetah in
:file:`demo_meta_matrixmul_cheetah.py` and
:file:`demo_meta_matrixmul_cheetah.template.cu`.

Metaprogramming using :mod:`codepy`
-----------------------------------

For more complicated metaprograms, it may be desirable to have more 
programmatic control over the assembly of the source code than a 
templating engine can provide. The :mod:`codepy` package provides a means
of generating CUDA source code from a Python data structure.

The following example demonstrates the use of :mod:`codepy` for 
metaprogramming. It accomplishes exactly the same as the above program::

    from codepy.cgen import FunctionBody, \
            FunctionDeclaration, Typedef, POD, Value, \
            Pointer, Module, Block, Initializer, Assign
    from codepy.cgen.cuda import CudaGlobal

    mod = Module([
        FunctionBody(
            CudaGlobal(FunctionDeclaration(
                Value("void", "add"),
                arg_decls=[Pointer(POD(dtype, name)) 
                    for name in ["tgt", "op1", "op2"]])),
            Block([
                Initializer(
                    POD(numpy.int32, "idx"),
                    "threadIdx.x + %d*blockIdx.x" 
                    % (thread_block_size*block_size)),
                ]+[
                Assign(
                    "tgt[idx+%d]" % (o*thread_block_size),
                    "op1[idx+%d] + op2[idx+%d]" % (
                        o*thread_block_size, 
                        o*thread_block_size))
                for o in range(block_size)]))])

    mod = SourceModule(mod)

This snippet in a working context can be found in 
:file:`examples/demo_meta_codepy.py`.
