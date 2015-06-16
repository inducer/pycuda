"""Scan primitive."""

from __future__ import division
from __future__ import absolute_import
import six

__copyright__ = """
Copyright 2011 Andreas Kloeckner
Copyright 2008-2011 NVIDIA Corporation
"""



__license__ = """
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Derived from thrust/detail/backend/cuda/detail/fast_scan.inl
within the Thrust project, https://code.google.com/p/thrust/

Direct browse link:
https://code.google.com/p/thrust/source/browse/thrust/detail/backend/cuda/detail/fast_scan.inl
"""




import numpy as np

import pycuda.driver as driver
import pycuda.gpuarray as gpuarray
from pycuda.compiler import SourceModule
from pycuda.tools import dtype_to_ctype
import pycuda._mymako as mako
from pycuda._cluda import CLUDA_PREAMBLE




SHARED_PREAMBLE = CLUDA_PREAMBLE + """
#define WG_SIZE ${wg_size}
#define SCAN_EXPR(a, b) ${scan_expr}

${preamble}

typedef ${scan_type} scan_type;
"""




SCAN_INTERVALS_SOURCE = mako.template.Template(SHARED_PREAMBLE + """//CL//
#define K ${wg_seq_batches}

<%def name="make_group_scan(name, with_bounds_check)">
    WITHIN_KERNEL
    void ${name}(LOCAL_MEM_ARG scan_type *array
    % if with_bounds_check:
      , const unsigned n
    % endif
    )
    {
        scan_type val = array[LID_0];

        <% offset = 1 %>

        % while offset <= wg_size:
            if (LID_0 >= ${offset}
            % if with_bounds_check:
              && LID_0 < n
            % endif
            )
            {
                scan_type tmp = array[LID_0 - ${offset}];
                val = SCAN_EXPR(tmp, val);
            }

            local_barrier();
            array[LID_0] = val;
            local_barrier();

            <% offset *= 2 %>
        % endwhile
    }
</%def>

${make_group_scan("scan_group", False)}
${make_group_scan("scan_group_n", True)}

KERNEL
REQD_WG_SIZE(WG_SIZE, 1, 1)
void ${name_prefix}_scan_intervals(
    GLOBAL_MEM scan_type *input,
    const unsigned int N,
    const unsigned int interval_size,
    GLOBAL_MEM scan_type *output,
    GLOBAL_MEM scan_type *group_results)
{
    // padded in WG_SIZE to avoid bank conflicts
    // index K in first dimension used for carry storage
    LOCAL_MEM scan_type ldata[K + 1][WG_SIZE + 1];

    const unsigned int interval_begin = interval_size * GID_0;
    const unsigned int interval_end   = min(interval_begin + interval_size, N);

    const unsigned int unit_size  = K * WG_SIZE;

    unsigned int unit_base = interval_begin;

    %for is_tail in [False, True]:

        %if not is_tail:
            for(; unit_base + unit_size <= interval_end; unit_base += unit_size)
        %else:
            if (unit_base < interval_end)
        %endif

        {
            // Algorithm: Each work group is responsible for one contiguous
            // 'interval', of which there are just enough to fill all compute
            // units.  Intervals are split into 'units'. A unit is what gets
            // worked on in parallel by one work group.

            // Each unit has two axes--the local-id axis and the k axis.
            //
            // * * * * * * * * * * ----> lid
            // * * * * * * * * * *
            // * * * * * * * * * *
            // * * * * * * * * * *
            // * * * * * * * * * *
            // |
            // v k

            // This is a three-phase algorithm, in which first each interval
            // does its local scan, then a scan across intervals exchanges data
            // globally, and the final update adds the exchanged sums to each
            // interval.

            // Exclusive scan is realized by performing a right-shift inside
            // the final update.

            // read a unit's worth of data from global

            for(unsigned int k = 0; k < K; k++)
            {
                const unsigned int offset = k*WG_SIZE + LID_0;

                %if is_tail:
                if (unit_base + offset < interval_end)
                %endif
                {
                    ldata[offset % K][offset / K] = input[unit_base + offset];
                }
            }

            // carry in from previous unit, if applicable.
            if (LID_0 == 0 && unit_base != interval_begin)
                ldata[0][0] = SCAN_EXPR(ldata[K][WG_SIZE - 1], ldata[0][0]);

            local_barrier();

            // scan along k (sequentially in each work item)
            scan_type sum = ldata[0][LID_0];

            %if is_tail:
                const unsigned int offset_end = interval_end - unit_base;
            %endif

            for(unsigned int k = 1; k < K; k++)
            {
                %if is_tail:
                if (K * LID_0 + k < offset_end)
                %endif
                {
                    scan_type tmp = ldata[k][LID_0];
                    sum = SCAN_EXPR(sum, tmp);
                    ldata[k][LID_0] = sum;
                }
            }

            // store carry in out-of-bounds (padding) array entry in the K direction
            ldata[K][LID_0] = sum;
            local_barrier();

            // tree-based parallel scan along local id
            %if not is_tail:
                scan_group(&ldata[K][0]);
            %else:
                scan_group_n(&ldata[K][0], offset_end / K);
            %endif

            // update local values
            if (LID_0 > 0)
            {
                sum = ldata[K][LID_0 - 1];

                for(unsigned int k = 0; k < K; k++)
                {
                    %if is_tail:
                    if (K * LID_0 + k < offset_end)
                    %endif
                    {
                        scan_type tmp = ldata[k][LID_0];
                        ldata[k][LID_0] = SCAN_EXPR(sum, tmp);
                    }
                }
            }

            local_barrier();

            // write data
            for(unsigned int k = 0; k < K; k++)
            {
                const unsigned int offset = k*WG_SIZE + LID_0;

                %if is_tail:
                if (unit_base + offset < interval_end)
                %endif
                {
                    output[unit_base + offset] = ldata[offset % K][offset / K];
                }
            }

            local_barrier();
        }

    % endfor

    // write interval sum
    if (LID_0 == 0)
    {
        group_results[GID_0] = output[interval_end - 1];
    }
}
""")




INCLUSIVE_UPDATE_SOURCE = mako.template.Template(SHARED_PREAMBLE + """//CL//
KERNEL
REQD_WG_SIZE(WG_SIZE, 1, 1)
void ${name_prefix}_final_update(
    GLOBAL_MEM scan_type *output,
    const unsigned int N,
    const unsigned int interval_size,
    GLOBAL_MEM scan_type *group_results)
{
    const unsigned int interval_begin = interval_size * GID_0;
    const unsigned int interval_end   = min(interval_begin + interval_size, N);

    if (GID_0 == 0)
        return;

    // value to add to this segment
    scan_type prev_group_sum = group_results[GID_0 - 1];

    // advance result pointer
    output += interval_begin + LID_0;

    for(unsigned int unit_base = interval_begin;
        unit_base < interval_end;
        unit_base += WG_SIZE, output += WG_SIZE)
    {
        const unsigned int i = unit_base + LID_0;

        if(i < interval_end)
        {
            *output = SCAN_EXPR(prev_group_sum, *output);
        }
    }
}
""")




EXCLUSIVE_UPDATE_SOURCE = mako.template.Template(SHARED_PREAMBLE + """//CL//
KERNEL
REQD_WG_SIZE(WG_SIZE, 1, 1)
void ${name_prefix}_final_update(
    GLOBAL_MEM scan_type *output,
    const unsigned int N,
    const unsigned int interval_size,
    GLOBAL_MEM scan_type *group_results)
{
    LOCAL_MEM scan_type ldata[WG_SIZE];

    const unsigned int interval_begin = interval_size * GID_0;
    const unsigned int interval_end   = min(interval_begin + interval_size, N);

    // value to add to this segment
    scan_type carry = ${neutral};
    if(GID_0 != 0)
    {
        scan_type tmp = group_results[GID_0 - 1];
        carry = SCAN_EXPR(carry, tmp);
    }

    scan_type val = carry;

    // advance result pointer
    output += interval_begin + LID_0;

    for (unsigned int unit_base = interval_begin;
        unit_base < interval_end;
        unit_base += WG_SIZE, output += WG_SIZE)
    {
        const unsigned int i = unit_base + LID_0;

        if(i < interval_end)
        {
            scan_type tmp = *output;
            ldata[LID_0] = SCAN_EXPR(carry, tmp);
        }

        local_barrier();

        if (LID_0 != 0)
            val = ldata[LID_0 - 1];
        /*
        else (see above)
            val = carry OR last tail;
        */

        if (i < interval_end)
            *output = val;

        if(LID_0 == 0)
            val = ldata[WG_SIZE - 1];

        local_barrier();
    }
}
""")




class _ScanKernelBase(object):
    def __init__(self, dtype,
            scan_expr, neutral=None,
            name_prefix="scan", options=[], preamble="", devices=None):

        if isinstance(self, ExclusiveScanKernel) and neutral is None:
            raise ValueError("neutral element is required for exclusive scan")

        dtype = self.dtype = np.dtype(dtype)
        self.neutral = neutral

        # Thrust says these are good for GT200
        self.scan_wg_size = 128
        self.update_wg_size = 256
        self.scan_wg_seq_batches = 6

        kw_values = dict(
            preamble=preamble,
            name_prefix=name_prefix,
            scan_type=dtype_to_ctype(dtype),
            scan_expr=scan_expr,
            neutral=neutral)

        scan_intervals_src = str(SCAN_INTERVALS_SOURCE.render(
            wg_size=self.scan_wg_size,
            wg_seq_batches=self.scan_wg_seq_batches,
            **kw_values))
        scan_intervals_prg = SourceModule(
                scan_intervals_src, options=options, no_extern_c=True)
        self.scan_intervals_knl = scan_intervals_prg.get_function(
                name_prefix+"_scan_intervals")
        self.scan_intervals_knl.prepare("PIIPP")

        final_update_src = str(self.final_update_tp.render(
            wg_size=self.update_wg_size,
            **kw_values))

        final_update_prg = SourceModule(
                final_update_src, options=options, no_extern_c=True)
        self.final_update_knl = final_update_prg.get_function(
                name_prefix+"_final_update")
        self.final_update_knl.prepare("PIIP")

    def __call__(self, input_ary, output_ary=None, allocator=None,
            stream=None):
        allocator = allocator or input_ary.allocator

        if output_ary is None:
            output_ary = input_ary

        if isinstance(output_ary, (str, six.text_type)) and output_ary == "new":
            output_ary = gpuarray.empty_like(input_ary, allocator=allocator)

        if input_ary.shape != output_ary.shape:
            raise ValueError("input and output must have the same shape")

        if not input_ary.flags.forc:
            raise RuntimeError("ScanKernel cannot "
                    "deal with non-contiguous arrays")

        n, = input_ary.shape

        if not n:
            return output_ary

        unit_size  = self.scan_wg_size * self.scan_wg_seq_batches
        dev = driver.Context.get_device()
        max_groups = 3*dev.get_attribute(
                driver.device_attribute.MULTIPROCESSOR_COUNT)

        from pytools import uniform_interval_splitting
        interval_size, num_groups = uniform_interval_splitting(
                n, unit_size, max_groups);

        block_results = allocator(self.dtype.itemsize*num_groups)
        dummy_results = allocator(self.dtype.itemsize)

        # first level scan of interval (one interval per block)
        self.scan_intervals_knl.prepared_async_call(
                (num_groups, 1), (self.scan_wg_size, 1, 1), stream,
                input_ary.gpudata,
                n, interval_size,
                output_ary.gpudata,
                block_results)

        # second level inclusive scan of per-block results
        self.scan_intervals_knl.prepared_async_call(
                (1,1), (self.scan_wg_size, 1, 1), stream,
                block_results,
                num_groups, interval_size,
                block_results,
                dummy_results)

        # update intervals with result of second level scan
        self.final_update_knl.prepared_async_call(
                (num_groups, 1,), (self.update_wg_size, 1, 1), stream,
                output_ary.gpudata,
                n, interval_size,
                block_results)

        return output_ary




class InclusiveScanKernel(_ScanKernelBase):
    final_update_tp = INCLUSIVE_UPDATE_SOURCE

class ExclusiveScanKernel(_ScanKernelBase):
    final_update_tp = EXCLUSIVE_UPDATE_SOURCE
