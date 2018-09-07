"""Scan primitive."""

from __future__ import division
from __future__ import absolute_import
import six
from six.moves import range, zip

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

Derived from code within the Thrust project, https://github.com/thrust/thrust/
"""

import numpy as np

import pycuda.driver as drv
import pycuda.gpuarray as gpuarray
from pycuda.compiler import SourceModule
from pycuda.tools import (dtype_to_ctype, bitlog2,
        _process_code_for_macro, get_arg_list_scalar_arg_dtypes,
        context_dependent_memoize,
        _NumpyTypesKeyBuilder)

import pycuda._mymako as mako
from pycuda._cluda import CLUDA_PREAMBLE

from pytools.persistent_dict import WriteOncePersistentDict


import logging
logger = logging.getLogger(__name__)

######################################################################################################
SHARED_PREAMBLE = CLUDA_PREAMBLE + """
#define WG_SIZE ${wg_size}

#define SCAN_EXPR(a, b, across_seg_boundary) ${scan_expr}
#define INPUT_EXPR(i) (${input_expr})
%if is_segmented:
    #define IS_SEG_START(i, a) (${is_segment_start_expr})
%endif

${preamble}

typedef ${dtype_to_ctype(scan_dtype)} scan_type;
typedef ${dtype_to_ctype(index_dtype)} index_type;

// NO_SEG_BOUNDARY is the largest representable integer in index_type.
// This assumption is used in code below.
#define NO_SEG_BOUNDARY ${str(np.iinfo(index_dtype).max)}
"""

# }}}

# {{{ main scan code

# Algorithm: Each work group is responsible for one contiguous
# 'interval'. There are just enough intervals to fill all compute
# units.  Intervals are split into 'units'. A unit is what gets
# worked on in parallel by one work group.
#
# in index space:
# interval > unit > local-parallel > k-group
#
# (Note that there is also a transpose in here: The data is read
# with local ids along linear index order.)
#
# Each unit has two axes--the local-id axis and the k axis.
#
# unit 0:
# | | | | | | | | | | ----> lid
# | | | | | | | | | |
# | | | | | | | | | |
# | | | | | | | | | |
# | | | | | | | | | |
#
# |
# v k (fastest-moving in linear index)
#
# unit 1:
# | | | | | | | | | | ----> lid
# | | | | | | | | | |
# | | | | | | | | | |
# | | | | | | | | | |
# | | | | | | | | | |
#
# |
# v k (fastest-moving in linear index)
#
# ...
#
# At a device-global level, this is a three-phase algorithm, in
# which first each interval does its local scan, then a scan
# across intervals exchanges data globally, and the final update
# adds the exchanged sums to each interval.
#
# Exclusive scan is realized by allowing look-behind (access to the
# preceding item) in the final update, by means of a local shift.
#
# NOTE: All segment_start_in_X indices are relative to the start
# of the array.

SCAN_INTERVALS_SOURCE = SHARED_PREAMBLE + r"""

#define K ${k_group_size}

// #define DEBUG
#ifdef DEBUG
    #define pycu_printf(ARGS) printf ARGS
#else
    #define pycu_printf(ARGS) /* */
#endif

KERNEL
REQD_WG_SIZE(WG_SIZE, 1, 1)
void ${kernel_name}(
    ${argument_signature},
    GLOBAL_MEM scan_type *partial_scan_buffer,
    const index_type N,
    const index_type interval_size
    %if is_first_level:
        , GLOBAL_MEM scan_type *interval_results
    %endif
    %if is_segmented and is_first_level:
        // NO_SEG_BOUNDARY if no segment boundary in interval.
        , GLOBAL_MEM index_type *g_first_segment_start_in_interval
    %endif
    %if store_segment_start_flags:
        , GLOBAL_MEM char *g_segment_start_flags
    %endif
    )
{
    // index K in first dimension used for carry storage
    %if use_bank_conflict_avoidance:
        // Avoid bank conflicts by adding a single 32-bit value to the size of
        // the scan type.
        struct __attribute__ ((__packed__)) wrapped_scan_type
        {
            scan_type value;
            int dummy;
        };
        LOCAL_MEM struct wrapped_scan_type ldata[K + 1][WG_SIZE + 1];
    %else:
        struct wrapped_scan_type
        {
            scan_type value;
        };

        // padded in WG_SIZE to avoid bank conflicts
        LOCAL_MEM struct wrapped_scan_type ldata[K + 1][WG_SIZE];
    %endif

    %if is_segmented:
        LOCAL_MEM char l_segment_start_flags[K][WG_SIZE];
        LOCAL_MEM index_type l_first_segment_start_in_subtree[WG_SIZE];

        // only relevant/populated for local id 0
        index_type first_segment_start_in_interval = NO_SEG_BOUNDARY;

        index_type first_segment_start_in_k_group, first_segment_start_in_subtree;
    %endif

    // {{{ declare local data for input_fetch_exprs if any of them are stenciled

    <%
        fetch_expr_offsets = {}
        for name, arg_name, ife_offset in input_fetch_exprs:
            fetch_expr_offsets.setdefault(arg_name, set()).add(ife_offset)

        local_fetch_expr_args = set(
            arg_name
            for arg_name, ife_offsets in fetch_expr_offsets.items()
            if -1 in ife_offsets or len(ife_offsets) > 1)
    %>

    %for arg_name in local_fetch_expr_args:
        LOCAL_MEM ${arg_ctypes[arg_name]} l_${arg_name}[WG_SIZE*K];
    %endfor

    // }}}

    const index_type interval_begin = interval_size * GID_0;
    const index_type interval_end   = min(interval_begin + interval_size, N);

    const index_type unit_size  = K * WG_SIZE;

    index_type unit_base = interval_begin;

    %for is_tail in [False, True]:

        %if not is_tail:
            for(; unit_base + unit_size <= interval_end; unit_base += unit_size)
        %else:
            if (unit_base < interval_end)
        %endif

        {

            // {{{ carry out input_fetch_exprs
            // (if there are ones that need to be fetched into local)

            %if local_fetch_expr_args:
                for(index_type k = 0; k < K; k++)
                {
                    const index_type offset = k*WG_SIZE + LID_0;
                    const index_type read_i = unit_base + offset;

                    %for arg_name in local_fetch_expr_args:
                        %if is_tail:
                        if (read_i < interval_end)
                        %endif
                        {
                            l_${arg_name}[offset] = ${arg_name}[read_i];
                        }
                    %endfor
                }

                local_barrier();
            %endif

            pycu_printf(("after input_fetch_exprs\n"));

            // }}}

            // {{{ read a unit's worth of data from global

            for(index_type k = 0; k < K; k++)
            {
                const index_type offset = k*WG_SIZE + LID_0;
                const index_type read_i = unit_base + offset;

                %if is_tail:
                if (read_i < interval_end)
                %endif
                {
                    %for name, arg_name, ife_offset in input_fetch_exprs:
                        ${arg_ctypes[arg_name]} ${name};

                        %if arg_name in local_fetch_expr_args:
                            if (offset + ${ife_offset} >= 0)
                                ${name} = l_${arg_name}[offset + ${ife_offset}];
                            else if (read_i + ${ife_offset} >= 0)
                                ${name} = ${arg_name}[read_i + ${ife_offset}];
                            /*
                            else
                                if out of bounds, name is left undefined */

                        %else:
                            // ${arg_name} gets fetched directly from global
                            ${name} = ${arg_name}[read_i];

                        %endif
                    %endfor

                    scan_type scan_value = INPUT_EXPR(read_i);

                    const index_type o_mod_k = offset % K;
                    const index_type o_div_k = offset / K;
                    ldata[o_mod_k][o_div_k].value = scan_value;

                    %if is_segmented:
                        bool is_seg_start = IS_SEG_START(read_i, scan_value);
                        l_segment_start_flags[o_mod_k][o_div_k] = is_seg_start;
                    %endif
                    %if store_segment_start_flags:
                        g_segment_start_flags[read_i] = is_seg_start;
                    %endif
                }
            }

            pycu_printf(("after read from global\n"));

            // }}}

            // {{{ carry in from previous unit, if applicable

            %if is_segmented:
                local_barrier();

                first_segment_start_in_k_group = NO_SEG_BOUNDARY;
                if (l_segment_start_flags[0][LID_0])
                    first_segment_start_in_k_group = unit_base + K*LID_0;
            %endif

            if (LID_0 == 0 && unit_base != interval_begin)
            {
                scan_type tmp = ldata[K][WG_SIZE - 1].value;
                scan_type tmp_aux = ldata[0][0].value;

                ldata[0][0].value = SCAN_EXPR(
                    tmp, tmp_aux,
                    %if is_segmented:
                        (l_segment_start_flags[0][0])
                    %else:
                        false
                    %endif
                    );
            }

            pycu_printf(("after carry-in\n"));

            // }}}

            local_barrier();

            // {{{ scan along k (sequentially in each work item)

            scan_type sum = ldata[0][LID_0].value;

            %if is_tail:
                const index_type offset_end = interval_end - unit_base;
            %endif

            for(index_type k = 1; k < K; k++)
            {
                %if is_tail:
                if (K * LID_0 + k < offset_end)
                %endif
                {
                    scan_type tmp = ldata[k][LID_0].value;

                    %if is_segmented:
                    index_type seq_i = unit_base + K*LID_0 + k;
                    if (l_segment_start_flags[k][LID_0])
                    {
                        first_segment_start_in_k_group = min(
                            first_segment_start_in_k_group,
                            seq_i);
                    }
                    %endif

                    sum = SCAN_EXPR(sum, tmp,
                        %if is_segmented:
                            (l_segment_start_flags[k][LID_0])
                        %else:
                            false
                        %endif
                        );

                    ldata[k][LID_0].value = sum;
                }
            }

            pycu_printf(("after scan along k\n"));

            // }}}

            // store carry in out-of-bounds (padding) array entry (index K) in
            // the K direction
            ldata[K][LID_0].value = sum;

            %if is_segmented:
                l_first_segment_start_in_subtree[LID_0] =
                    first_segment_start_in_k_group;
            %endif

            local_barrier();

            // {{{ tree-based local parallel scan

            // This tree-based scan works as follows:
            // - Each work item adds the previous item to its current state
            // - barrier
            // - Each work item adds in the item from two positions to the left
            // - barrier
            // - Each work item adds in the item from four positions to the left
            // ...
            // At the end, each item has summed all prior items.

            // across k groups, along local id
            // (uses out-of-bounds k=K array entry for storage)

            scan_type val = ldata[K][LID_0].value;

            <% scan_offset = 1 %>

            % while scan_offset <= wg_size:
                // {{{ reads from local allowed, writes to local not allowed

                if (LID_0 >= ${scan_offset})
                {
                    scan_type tmp = ldata[K][LID_0 - ${scan_offset}].value;
                    % if is_tail:
                    if (K*LID_0 < offset_end)
                    % endif
                    {
                        val = SCAN_EXPR(tmp, val,
                            %if is_segmented:
                                (l_first_segment_start_in_subtree[LID_0]
                                    != NO_SEG_BOUNDARY)
                            %else:
                                false
                            %endif
                            );
                    }

                    %if is_segmented:
                        // Prepare for l_first_segment_start_in_subtree, below.

                        // Note that this update must take place *even* if we're
                        // out of bounds.

                        first_segment_start_in_subtree = min(
                            l_first_segment_start_in_subtree[LID_0],
                            l_first_segment_start_in_subtree
                                [LID_0 - ${scan_offset}]);
                    %endif
                }
                %if is_segmented:
                    else
                    {
                        first_segment_start_in_subtree =
                            l_first_segment_start_in_subtree[LID_0];
                    }
                %endif

                // }}}

                local_barrier();

                // {{{ writes to local allowed, reads from local not allowed

                ldata[K][LID_0].value = val;
                %if is_segmented:
                    l_first_segment_start_in_subtree[LID_0] =
                        first_segment_start_in_subtree;
                %endif

                // }}}

                local_barrier();

                %if 0:
                if (LID_0 == 0)
                {
                    printf("${scan_offset}: ");
                    for (int i = 0; i < WG_SIZE; ++i)
                    {
                        if (l_first_segment_start_in_subtree[i] == NO_SEG_BOUNDARY)
                            printf("- ");
                        else
                            printf("%d ", l_first_segment_start_in_subtree[i]);
                    }
                    printf("\n");
                }
                %endif

                <% scan_offset *= 2 %>
            % endwhile

            pycu_printf(("after tree scan\n"));

            // }}}

            // {{{ update local values

            if (LID_0 > 0)
            {
                sum = ldata[K][LID_0 - 1].value;

                for(index_type k = 0; k < K; k++)
                {
                    %if is_tail:
                    if (K * LID_0 + k < offset_end)
                    %endif
                    {
                        scan_type tmp = ldata[k][LID_0].value;
                        ldata[k][LID_0].value = SCAN_EXPR(sum, tmp,
                            %if is_segmented:
                                (unit_base + K * LID_0 + k
                                    >= first_segment_start_in_k_group)
                            %else:
                                false
                            %endif
                            );
                    }
                }
            }

            %if is_segmented:
                if (LID_0 == 0)
                {
                    // update interval-wide first-seg variable from current unit
                    first_segment_start_in_interval = min(
                        first_segment_start_in_interval,
                        l_first_segment_start_in_subtree[WG_SIZE-1]);
                }
            %endif

            pycu_printf(("after local update\n"));

            // }}}

            local_barrier();

            // {{{ write data

            {
                // work hard with index math to achieve contiguous 32-bit stores
                GLOBAL_MEM int *dest =
                    (GLOBAL_MEM int *) (partial_scan_buffer + unit_base);

                <%

                assert scan_dtype.itemsize % 4 == 0

                ints_per_wg = wg_size
                ints_to_store = scan_dtype.itemsize*wg_size*k_group_size // 4

                %>

                const index_type scan_types_per_int = ${scan_dtype.itemsize//4};

                %for store_base in range(0, ints_to_store, ints_per_wg):
                    <%

                    # Observe that ints_to_store is divisible by the work group
                    # size already, so we won't go out of bounds that way.
                    assert store_base + ints_per_wg <= ints_to_store

                    %>

                    %if is_tail:
                    if (${store_base} + LID_0 <
                        scan_types_per_int*(interval_end - unit_base))
                    %endif
                    {
                        index_type linear_index = ${store_base} + LID_0;
                        index_type linear_scan_data_idx =
                            linear_index / scan_types_per_int;
                        index_type remainder =
                            linear_index - linear_scan_data_idx * scan_types_per_int;

                        dest[linear_index] = ( &(ldata
                                [linear_scan_data_idx % K]
                                [linear_scan_data_idx / K].value))[remainder];
                    }
                %endfor
            }
            pycu_printf(("after write\n"));

            // }}}

            local_barrier();
        }

    % endfor

    // write interval sum
    %if is_first_level:
        if (LID_0 == 0)
        {
            interval_results[GID_0] = partial_scan_buffer[interval_end - 1];
            %if is_segmented:
                g_first_segment_start_in_interval[GID_0] =
                    first_segment_start_in_interval;
            %endif
        }
    %endif
}
"""

# }}}

# {{{ update

UPDATE_SOURCE = SHARED_PREAMBLE + r"""

KERNEL
REQD_WG_SIZE(WG_SIZE, 1, 1)
void ${name_prefix}_final_update(
    ${argument_signature},
    const index_type N,
    const index_type interval_size,
    GLOBAL_MEM scan_type *interval_results,
    GLOBAL_MEM scan_type *partial_scan_buffer
    %if is_segmented:
        , GLOBAL_MEM index_type *g_first_segment_start_in_interval
    %endif
    %if is_segmented and use_lookbehind_update:
        , GLOBAL_MEM char *g_segment_start_flags
    %endif
    )
{
    %if use_lookbehind_update:
        LOCAL_MEM scan_type ldata[WG_SIZE];
    %endif
    %if is_segmented and use_lookbehind_update:
        LOCAL_MEM char l_segment_start_flags[WG_SIZE];
    %endif

    const index_type interval_begin = interval_size * GID_0;
    const index_type interval_end = min(interval_begin + interval_size, N);

    // carry from last interval
    scan_type carry = ${neutral};
    if (GID_0 != 0)
        carry = interval_results[GID_0 - 1];

    %if is_segmented:
        const index_type first_seg_start_in_interval =
            g_first_segment_start_in_interval[GID_0];
    %endif

    %if not is_segmented and 'last_item' in output_statement:
        scan_type last_item = interval_results[GDIM_0-1];
    %endif

    %if not use_lookbehind_update:
        // {{{ no look-behind ('prev_item' not in output_statement -> simpler)

        index_type update_i = interval_begin+LID_0;

        %if is_segmented:
            index_type seg_end = min(first_seg_start_in_interval, interval_end);
        %endif

        for(; update_i < interval_end; update_i += WG_SIZE)
        {
            scan_type partial_val = partial_scan_buffer[update_i];
            scan_type item = SCAN_EXPR(carry, partial_val,
                %if is_segmented:
                    (update_i >= seg_end)
                %else:
                    false
                %endif
                );
            index_type i = update_i;

            { ${output_statement}; }
        }

        // }}}
    %else:
        // {{{ allow look-behind ('prev_item' in output_statement -> complicated)

        // We are not allowed to branch across barriers at a granularity smaller
        // than the whole workgroup. Therefore, the for loop is group-global,
        // and there are lots of local ifs.

        index_type group_base = interval_begin;
        scan_type prev_item = carry; // (A)

        for(; group_base < interval_end; group_base += WG_SIZE)
        {
            index_type update_i = group_base+LID_0;

            // load a work group's worth of data
            if (update_i < interval_end)
            {
                scan_type tmp = partial_scan_buffer[update_i];

                tmp = SCAN_EXPR(carry, tmp,
                    %if is_segmented:
                        (update_i >= first_seg_start_in_interval)
                    %else:
                        false
                    %endif
                    );

                ldata[LID_0] = tmp;

                %if is_segmented:
                    l_segment_start_flags[LID_0] = g_segment_start_flags[update_i];
                %endif
            }

            local_barrier();

            // find prev_item
            if (LID_0 != 0)
                prev_item = ldata[LID_0 - 1];
            /*
            else
                prev_item = carry (see (A)) OR last tail (see (B));
            */

            if (update_i < interval_end)
            {
                %if is_segmented:
                    if (l_segment_start_flags[LID_0])
                        prev_item = ${neutral};
                %endif

                scan_type item = ldata[LID_0];
                index_type i = update_i;
                { ${output_statement}; }
            }

            if (LID_0 == 0)
                prev_item = ldata[WG_SIZE - 1]; // (B)

            local_barrier();
        }

        // }}}
    %endif
}
"""

# }}}

# {{{ driver

# {{{ helpers

def _round_down_to_power_of_2(val):
    result = 2**bitlog2(val)
    if result > val:
        result >>= 1

    assert result <= val
    return result


_PREFIX_WORDS = set("""
        ldata partial_scan_buffer global scan_offset
        segment_start_in_k_group carry
        g_first_segment_start_in_interval IS_SEG_START tmp Z
        val l_first_segment_start_in_subtree unit_size
        index_type interval_begin interval_size offset_end K
        SCAN_EXPR do_update WG_SIZE
        first_segment_start_in_k_group scan_type
        segment_start_in_subtree offset interval_results interval_end
        first_segment_start_in_subtree unit_base
        first_segment_start_in_interval k INPUT_EXPR
        prev_group_sum prev pv value partial_val pgs
        is_seg_start update_i scan_item_at_i seq_i read_i
        l_ o_mod_k o_div_k l_segment_start_flags scan_value sum
        first_seg_start_in_interval g_segment_start_flags
        group_base seg_end my_val DEBUG ARGS
        ints_to_store ints_per_wg scan_types_per_int linear_index
        linear_scan_data_idx dest src store_base wrapped_scan_type
        dummy scan_tmp tmp_aux

        LID_2 LID_1 LID_0
        LDIM_0 LDIM_1 LDIM_2
        GDIM_0 GDIM_1 GDIM_2
        GID_0 GID_1 GID_2
        """.split())

_IGNORED_WORDS = set("""
        4 8 32

        typedef for endfor if void while endwhile endfor endif else const printf
        None return bool n char true false ifdef pycu_printf str range assert
        np iinfo max itemsize __packed__ struct

        set iteritems len setdefault

        GLOBAL_MEM LOCAL_MEM_ARG WITHIN_KERNEL LOCAL_MEM KERNEL REQD_WG_SIZE
        local_barrier
        CLK_LOCAL_MEM_FENCE
        pragma __attribute__
        get_local_size get_local_id cl_khr_fp64 reqd_work_group_size
        get_num_groups barrier get_group_id

        _final_update _debug_scan kernel_name

        positions all padded integer its previous write based writes 0
        has local worth scan_expr to read cannot not X items False bank
        four beginning follows applicable item min each indices works side
        scanning right summed relative used id out index avoid current state
        boundary True across be This reads groups along Otherwise undetermined
        store of times prior s update first regardless Each number because
        array unit from segment conflicts two parallel 2 empty define direction
        CL padding work tree bounds values and adds
        scan is allowed thus it an as enable at in occur sequentially end no
        storage data 1 largest may representable uses entry Y meaningful
        computations interval At the left dimension know d
        A load B group perform shift tail see last OR
        this add fetched into are directly need
        gets them stenciled that undefined
        there up any ones or name only relevant populated
        even wide we Prepare int seg Note re below place take variable must
        intra Therefore find code assumption
        branch workgroup complicated granularity phase remainder than simpler
        We smaller look ifs lots self behind allow barriers whole loop
        after already Observe achieve contiguous stores hard go with by math
        size won t way divisible bit so Avoid declare adding single type

        is_tail is_first_level input_expr argument_signature preamble
        double_support neutral output_statement
        k_group_size name_prefix is_segmented index_dtype scan_dtype
        wg_size is_segment_start_expr fetch_expr_offsets
        arg_ctypes ife_offsets input_fetch_exprs def
        ife_offset arg_name local_fetch_expr_args update_body
        update_loop_lookbehind update_loop_plain update_loop
        use_lookbehind_update store_segment_start_flags
        update_loop first_seg scan_dtype dtype_to_ctype
        use_bank_conflict_avoidance

        a b prev_item i last_item prev_value
        N NO_SEG_BOUNDARY across_seg_boundary
        """.split())


def _make_template(s):
    leftovers = set()

    def replace_id(match):
        # avoid name clashes with user code by adding 'psc_' prefix to
        # identifiers.

        word = match.group(1)
        if word in _IGNORED_WORDS:
            return word
        elif word in _PREFIX_WORDS:
            return "psc_"+word
        else:
            leftovers.add(word)
            return word

    import re
    s = re.sub(r"\b([a-zA-Z0-9_]+)\b", replace_id, s)

    if leftovers:
        from warnings import warn
        warn("leftover words in identifier prefixing: " + " ".join(leftovers))

    return mako.template.Template(s, strict_undefined=True)


from pytools import Record, RecordWithoutPickling


class _GeneratedScanKernelInfo(Record):

    __slots__ = [
            "scan_src",
            "kernel_name",
            "scalar_arg_dtypes",
            "wg_size",
            "k_group_size"]

    def __init__(self, scan_src, kernel_name, scalar_arg_dtypes, wg_size,
            k_group_size):
        Record.__init__(self,
                scan_src=scan_src,
                kernel_name=kernel_name,
                scalar_arg_dtypes=scalar_arg_dtypes,
                wg_size=wg_size,
                k_group_size=k_group_size)

    def build(self, options):
        program = SourceModule(self.scan_src, options=options)
        kernel = program.get_function(self.kernel_name)
        kernel.prepare(self.scalar_arg_dtypes)
        return _BuiltScanKernelInfo(
                kernel=kernel,
                wg_size=self.wg_size,
                k_group_size=self.k_group_size)


class _BuiltScanKernelInfo(RecordWithoutPickling):

    __slots__ = ["kernel", "wg_size", "k_group_size"]

    def __init__(self, kernel, wg_size, k_group_size):
        RecordWithoutPickling.__init__(self,
                kernel=kernel,
                wg_size=wg_size,
                k_group_size=k_group_size)


class _GeneratedFinalUpdateKernelInfo(Record):

    def __init__(self, source, kernel_name, scalar_arg_dtypes, update_wg_size):
        Record.__init__(self,
                source=source,
                kernel_name=kernel_name,
                scalar_arg_dtypes=scalar_arg_dtypes,
                update_wg_size=update_wg_size)

    def build(self, options):
        program = SourceModule(self.source, options=options)
        kernel = program.get_function(self.kernel_name)
        kernel.prepare(self.scalar_arg_dtypes)
        return _BuiltFinalUpdateKernelInfo(
                kernel=kernel,
                update_wg_size=self.update_wg_size
                )


class _BuiltFinalUpdateKernelInfo(RecordWithoutPickling):
    __slots__ = ["kernel", "update_wg_size"]

    def __init__(self, kernel, update_wg_size):
        RecordWithoutPickling.__init__(self,
                kernel=kernel,
                update_wg_size=update_wg_size)

# }}}


class ScanPerformanceWarning(UserWarning):
    pass


class _GenericScanKernelBase(object):
    # {{{ constructor, argument processing

    def __init__(self, dtype,
            arguments, input_expr, scan_expr, neutral, output_statement,
            is_segment_start_expr=None, input_fetch_exprs=[],
            index_dtype=np.int32,
            name_prefix="scan", options=[], preamble=""):
        """
        :arg dtype: the :class:`numpy.dtype` with which the scan will
            be performed. May be a structured type if that type was registered
            through :func:`pyopencl.tools.get_or_register_dtype`.
        :arg arguments: A string of comma-separated C argument declarations.
            If *arguments* is specified, then *input_expr* must also be
            specified. All types used here must be known to PyCUDA.
            (see :func:`pycuda.tools.get_or_register_dtype`).
        :arg scan_expr: The associative, binary operation carrying out the scan,
            represented as a C string. Its two arguments are available as `a`
            and `b` when it is evaluated. `b` is guaranteed to be the
            'element being updated', and `a` is the increment. Thus,
            if some data is supposed to just propagate along without being
            modified by the scan, it should live in `b`.

            This expression may call functions given in the *preamble*.

            Another value available to this expression is `across_seg_boundary`,
            a C `bool` indicating whether this scan update is crossing a
            segment boundary, as defined by `is_segment_start_expr`.
            The scan routine does not implement segmentation
            semantics on its own. It relies on `scan_expr` to do this.
            This value is available (but always `false`) even for a
            non-segmented scan.

            .. note::

                In early pre-releases of the segmented scan,
                segmentation semantics were implemented *without*
                relying on `scan_expr`.

        :arg input_expr: A C expression, encoded as a string, resulting
            in the values to which the scan is applied. This may be used
            to apply a mapping to values stored in *arguments* before being
            scanned. The result of this expression must match *dtype*.
            The index intended to be mapped is available as `i` in this
            expression. This expression may also use the variables defined
            by *input_fetch_expr*.

            This expression may also call functions given in the *preamble*.
        :arg output_statement: a C statement that writes
            the output of the scan. It has access to the scan result as `item`,
            the preceding scan result item as `prev_item`, and the current index
            as `i`. `prev_item` in a segmented scan will be the neutral element
            at a segment boundary, not the immediately preceding item.

            Using *prev_item* in output statement has a small run-time cost.
            `prev_item` enables the construction of an exclusive scan.

            For non-segmented scans, *output_statement* may also reference
            `last_item`, which evaluates to the scan result of the last
            array entry.
        :arg is_segment_start_expr: A C expression, encoded as a string,
            resulting in a C `bool` value that determines whether a new
            scan segments starts at index *i*.  If given, makes the scan a
            segmented scan. Has access to the current index `i`, the result
            of *input_expr* as a, and in addition may use *arguments* and
            *input_fetch_expr* variables just like *input_expr*.

            If it returns true, then previous sums will not spill over into the
            item with index *i* or subsequent items.
        :arg input_fetch_exprs: a list of tuples *(NAME, ARG_NAME, OFFSET)*.
            An entry here has the effect of doing the equivalent of the following
            before input_expr::

                ARG_NAME_TYPE NAME = ARG_NAME[i+OFFSET];

            `OFFSET` is allowed to be 0 or -1, and `ARG_NAME_TYPE` is the type
            of `ARG_NAME`.
        :arg preamble: |preamble|

        The first array in the argument list determines the size of the index
        space over which the scan is carried out, and thus the values over
        which the index *i* occurring in a number of code fragments in
        arguments above will vary.

        All code fragments further have access to N, the number of elements
        being processed in the scan.
        """

        dtype = self.dtype = np.dtype(dtype)

        if neutral is None:
            from warnings import warn
            warn("not specifying 'neutral' is deprecated and will lead to "
                    "wrong results if your scan is not in-place or your "
                    "'output_statement' does something otherwise non-trivial",
                    stacklevel=2)

        if dtype.itemsize % 4 != 0:
            raise TypeError("scan value type must have size divisible by 4 bytes")

        self.index_dtype = np.dtype(index_dtype)
        if np.iinfo(self.index_dtype).min >= 0:
            raise TypeError("index_dtype must be signed")

        self.options = options

        from pycuda.tools import parse_arg_list
        self.parsed_args = parse_arg_list(arguments)
        from pycuda.tools import VectorArg
        vector_args_indices = [i for i, arg in enumerate(self.parsed_args) \
                if isinstance(arg, VectorArg)]
        self.first_array_idx = vector_args_indices[0]

        self.input_expr = input_expr

        self.is_segment_start_expr = is_segment_start_expr
        self.is_segmented = is_segment_start_expr is not None
        if self.is_segmented:
            is_segment_start_expr = _process_code_for_macro(is_segment_start_expr)

        self.output_statement = output_statement

        for name, arg_name, ife_offset in input_fetch_exprs:
            if ife_offset not in [0, -1]:
                raise RuntimeError("input_fetch_expr offsets must either be 0 or -1")
        self.input_fetch_exprs = input_fetch_exprs

        arg_dtypes = {}
        arg_ctypes = {}
        for arg in self.parsed_args:
            arg_dtypes[arg.name] = arg.dtype
            arg_ctypes[arg.name] = dtype_to_ctype(arg.dtype)

        self.options = options
        self.name_prefix = name_prefix

        # {{{ set up shared code dict

        from pytools import all
        from pycuda.characterize import has_double_support

        self.code_variables = dict(
            np=np,
            dtype_to_ctype=dtype_to_ctype,
            preamble=preamble,
            name_prefix=name_prefix,
            index_dtype=self.index_dtype,
            scan_dtype=dtype,
            is_segmented=self.is_segmented,
            arg_dtypes=arg_dtypes,
            arg_ctypes=arg_ctypes,
            scan_expr=_process_code_for_macro(scan_expr),
            neutral=_process_code_for_macro(neutral),
            double_support=has_double_support(),
            )

        index_typename = dtype_to_ctype(self.index_dtype)
        scan_typename = dtype_to_ctype(dtype)

        # This key is meant to uniquely identify the non-device parameters for
        # the scan kernel.
        self.kernel_key = (
            self.dtype,
            tuple(arg.declarator() for arg in self.parsed_args),
            self.input_expr,
            scan_expr,
            neutral,
            output_statement,
            is_segment_start_expr,
            tuple(input_fetch_exprs),
            index_dtype,
            name_prefix,
            preamble,
            # These depend on dtype_to_ctype(), so their value is independent of
            # the other variables.
            index_typename,
            scan_typename,
            )

        # }}}

        self.use_lookbehind_update = "prev_item" in self.output_statement
        self.store_segment_start_flags = (
                self.is_segmented and self.use_lookbehind_update)

        self.finish_setup()

    # }}}


generic_scan_kernel_cache = WriteOncePersistentDict(
        "pycuda-generated-scan-kernel-cache-v1",
        key_builder=_NumpyTypesKeyBuilder())


class GenericScanKernel(_GenericScanKernelBase):
    """Generates and executes code that performs prefix sums ("scans") on
    arbitrary types, with many possible tweaks.

    Usage example::

        import pycuda as cu
        from pycuda.scan import GenericScanKernel
        knl = GenericScanKernel(
                np.int32,
                arguments="int *ary",
                input_expr="ary[i]",
                scan_expr="a+b", neutral="0",
                output_statement="ary[i+1] = item;")

        a = cu.gpuarray.arange(10000, dtype=np.int32)
        knl(a)

    """

    def finish_setup(self):
        # Before generating the kernel, see if it's cached.

        cache_key = (self.kernel_key,)

        from_cache = False

        try:
            result = generic_scan_kernel_cache[cache_key]
            from_cache = True
            logger.debug(
                    "cache hit for generated scan kernel '%s'" % self.name_prefix)
            (
                self.first_level_scan_gen_info,
                self.second_level_scan_gen_info,
                self.final_update_gen_info) = result
        except KeyError:
            pass

        if not from_cache:
            logger.debug(
                    "cache miss for generated scan kernel '%s'" % self.name_prefix)
            self._finish_setup_impl()

            result = (self.first_level_scan_gen_info,
                      self.second_level_scan_gen_info,
                      self.final_update_gen_info)

            generic_scan_kernel_cache.store_if_not_present(cache_key, result)

        # Build the kernels.
        self.first_level_scan_info = self.first_level_scan_gen_info.build(
                self.options)
        del self.first_level_scan_gen_info

        self.second_level_scan_info = self.second_level_scan_gen_info.build(
                self.options)
        del self.second_level_scan_gen_info

        self.final_update_info = self.final_update_gen_info.build(
                self.options)
        del self.final_update_gen_info

    def _finish_setup_impl(self):
        # {{{ find usable workgroup/k-group size, build first-level scan

        trip_count = 0

        dev = drv.Context.get_device()
        avail_local_mem = dev.get_attribute(
                 drv.device_attribute.MAX_SHARED_MEMORY_PER_BLOCK)

        # not sure where these go, but roughly this much seems unavailable.
        avail_local_mem -= 0x400

        max_scan_wg_size = dev.get_attribute(
                drv.device_attribute.MAX_THREADS_PER_BLOCK)
        wg_size_multiples = 64

        use_bank_conflict_avoidance = (
                self.dtype.itemsize > 4 and self.dtype.itemsize % 8 == 0)

        # k_group_size should be a power of two because of in-kernel
        # division by that number.

        solutions = []
        for k_exp in range(0, 9):
            for wg_size in range(wg_size_multiples, max_scan_wg_size+1,
                    wg_size_multiples):

                k_group_size = 2**k_exp
                lmem_use = self.get_local_mem_use(wg_size, k_group_size,
                        use_bank_conflict_avoidance)
                if lmem_use <= avail_local_mem:
                    solutions.append((wg_size*k_group_size, k_group_size, wg_size))

        from pytools import any
        for wg_size_floor in [256, 192, 128]:
            have_sol_above_floor = any(wg_size >= wg_size_floor
                    for _, _, wg_size in solutions)

            if have_sol_above_floor:
                # delete all solutions not meeting the wg size floor
                solutions = [(total, try_k_group_size, try_wg_size)
                        for total, try_k_group_size, try_wg_size in solutions
                        if try_wg_size >= wg_size_floor]
                break

        _, k_group_size, max_scan_wg_size = max(solutions)

        while True:
            candidate_scan_gen_info = self.generate_scan_kernel(
                    max_scan_wg_size, self.parsed_args,
                    _process_code_for_macro(self.input_expr),
                    self.is_segment_start_expr,
                    input_fetch_exprs=self.input_fetch_exprs,
                    is_first_level=True,
                    store_segment_start_flags=self.store_segment_start_flags,
                    k_group_size=k_group_size,
                    use_bank_conflict_avoidance=use_bank_conflict_avoidance)

            candidate_scan_info = candidate_scan_gen_info.build(
                    self.options)

            # Will this device actually let us execute this kernel
            # at the desired work group size? Building it is the
            # only way to find out.
            kernel_max_wg_size = candidate_scan_info.kernel.get_attribute(
                    drv.function_attribute.MAX_THREADS_PER_BLOCK)

            if candidate_scan_info.wg_size <= kernel_max_wg_size:
                break
            else:
                max_scan_wg_size = min(kernel_max_wg_size, max_scan_wg_size)

            trip_count += 1
            assert trip_count <= 20

        self.first_level_scan_gen_info = candidate_scan_gen_info
        assert (_round_down_to_power_of_2(candidate_scan_info.wg_size)
                == candidate_scan_info.wg_size)

        # }}}

        # {{{ build second-level scan

        from pycuda.tools import VectorArg
        second_level_arguments = self.parsed_args + [
                VectorArg(self.dtype, "interval_sums")]

        second_level_build_kwargs = {}
        if self.is_segmented:
            second_level_arguments.append(
                    VectorArg(self.index_dtype,
                        "g_first_segment_start_in_interval_input"))

            # is_segment_start_expr answers the question "should previous sums
            # spill over into this item". And since
            # g_first_segment_start_in_interval_input answers the question if a
            # segment boundary was found in an interval of data, then if not,
            # it's ok to spill over.
            second_level_build_kwargs["is_segment_start_expr"] = \
                    "g_first_segment_start_in_interval_input[i] != NO_SEG_BOUNDARY"
        else:
            second_level_build_kwargs["is_segment_start_expr"] = None

        self.second_level_scan_gen_info = self.generate_scan_kernel(
                max_scan_wg_size,
                arguments=second_level_arguments,
                input_expr="interval_sums[i]",
                input_fetch_exprs=[],
                is_first_level=False,
                store_segment_start_flags=False,
                k_group_size=k_group_size,
                use_bank_conflict_avoidance=use_bank_conflict_avoidance,
                **second_level_build_kwargs)

        # }}}

        # {{{ generate final update kernel

        update_wg_size = min(max_scan_wg_size, 256)

        final_update_tpl = _make_template(UPDATE_SOURCE)
        final_update_src = str(final_update_tpl.render(
            wg_size=update_wg_size,
            output_statement=self.output_statement,
            argument_signature=", ".join(
                arg.declarator() for arg in self.parsed_args),
            is_segment_start_expr=self.is_segment_start_expr,
            input_expr=_process_code_for_macro(self.input_expr),
            use_lookbehind_update=self.use_lookbehind_update,
            **self.code_variables))

        update_scalar_arg_dtypes = (
                get_arg_list_scalar_arg_dtypes(self.parsed_args)
                + [self.index_dtype, self.index_dtype, None, None])
        if self.is_segmented:
            # g_first_segment_start_in_interval
            update_scalar_arg_dtypes.append(None)
        if self.store_segment_start_flags:
            update_scalar_arg_dtypes.append(None)  # g_segment_start_flags

        self.final_update_gen_info = _GeneratedFinalUpdateKernelInfo(
                final_update_src,
                self.name_prefix + "_final_update",
                update_scalar_arg_dtypes,
                update_wg_size)

        # }}}

    # {{{ scan kernel build/properties

    def get_local_mem_use(self, k_group_size, wg_size,
            use_bank_conflict_avoidance):
        arg_dtypes = {}
        for arg in self.parsed_args:
            arg_dtypes[arg.name] = arg.dtype

        fetch_expr_offsets = {}
        for name, arg_name, ife_offset in self.input_fetch_exprs:
            fetch_expr_offsets.setdefault(arg_name, set()).add(ife_offset)

        itemsize = self.dtype.itemsize
        if use_bank_conflict_avoidance:
            itemsize += 4

        return (
                # ldata
                itemsize*(k_group_size+1)*(wg_size+1)

                # l_segment_start_flags
                + k_group_size*wg_size

                # l_first_segment_start_in_subtree
                + self.index_dtype.itemsize*wg_size

                + k_group_size*wg_size*sum(
                    arg_dtypes[arg_name].itemsize
                    for arg_name, ife_offsets in list(fetch_expr_offsets.items())
                    if -1 in ife_offsets or len(ife_offsets) > 1))

    def generate_scan_kernel(self, max_wg_size, arguments, input_expr,
            is_segment_start_expr, input_fetch_exprs, is_first_level,
            store_segment_start_flags, k_group_size,
            use_bank_conflict_avoidance):
        scalar_arg_dtypes = get_arg_list_scalar_arg_dtypes(arguments)

        # Empirically found on Nv hardware: no need to be bigger than this size
        wg_size = _round_down_to_power_of_2(
                min(max_wg_size, 256))

        kernel_name = self.code_variables["name_prefix"]
        if is_first_level:
            kernel_name += "_lev1"
        else:
            kernel_name += "_lev2"

        scan_tpl = _make_template(SCAN_INTERVALS_SOURCE)
        scan_src = str(scan_tpl.render(
            wg_size=wg_size,
            input_expr=input_expr,
            k_group_size=k_group_size,
            argument_signature=", ".join(arg.declarator() for arg in arguments),
            is_segment_start_expr=is_segment_start_expr,
            input_fetch_exprs=input_fetch_exprs,
            is_first_level=is_first_level,
            store_segment_start_flags=store_segment_start_flags,
            use_bank_conflict_avoidance=use_bank_conflict_avoidance,
            kernel_name=kernel_name,
            **self.code_variables))

        scalar_arg_dtypes.extend(
                (None, self.index_dtype, self.index_dtype))
        if is_first_level:
            scalar_arg_dtypes.append(None)  # interval_results
        if self.is_segmented and is_first_level:
            scalar_arg_dtypes.append(None)  # g_first_segment_start_in_interval
        if store_segment_start_flags:
            scalar_arg_dtypes.append(None)  # g_segment_start_flags

        return _GeneratedScanKernelInfo(
                scan_src=scan_src,
                kernel_name=kernel_name,
                scalar_arg_dtypes=scalar_arg_dtypes,
                wg_size=wg_size,
                k_group_size=k_group_size)

    # }}}

    def __call__(self, *args, **kwargs):
        # {{{ argument processing

        allocator = kwargs.get("allocator")
        n = kwargs.get("size")
        stream = kwargs.get("stream")

        if len(args) != len(self.parsed_args):
            raise TypeError("expected %d arguments, got %d" %
                    (len(self.parsed_args), len(args)))

        first_array = args[self.first_array_idx]
        allocator = allocator or first_array.allocator

        if n is None:
            n, = first_array.shape

        if n == 0:
            return

        data_args = []
        from pycuda.tools import VectorArg
        for arg_descr, arg_val in zip(self.parsed_args, args):
            if isinstance(arg_descr, VectorArg):
                data_args.append(arg_val.gpudata)
            else:
                data_args.append(arg_val)

        # }}}

        l1_info = self.first_level_scan_info
        l2_info = self.second_level_scan_info

        unit_size = l1_info.wg_size * l1_info.k_group_size
        dev = drv.Context.get_device()
        max_intervals = 3*dev.get_attribute(
                 drv.device_attribute.MULTIPROCESSOR_COUNT)

        from pytools import uniform_interval_splitting
        interval_size, num_intervals = uniform_interval_splitting(
                n, unit_size, max_intervals)

        # {{{ allocate some buffers

        interval_results = gpuarray.empty(
                num_intervals, dtype=self.dtype,
                allocator=allocator)

        partial_scan_buffer = gpuarray.empty(
                n, dtype=self.dtype,
                allocator=allocator)

        if self.store_segment_start_flags:
            segment_start_flags = gpuarray.empty(
                    n, dtype=np.bool,
                    allocator=allocator)

        # }}}

        # {{{ first level scan of interval (one interval per block)

        scan1_args = data_args + [
                partial_scan_buffer.gpudata, n, interval_size,
                interval_results.gpudata,
                ]

        if self.is_segmented:
            first_segment_start_in_interval = gpuarray.empty(
                    num_intervals, dtype=self.index_dtype,
                    allocator=allocator)
            scan1_args.append(first_segment_start_in_interval.gpudata)

        if self.store_segment_start_flags:
            scan1_args.append(segment_start_flags.gpudata)

        l1_evt = l1_info.kernel.prepared_async_call(
                (num_intervals, 1), (l1_info.wg_size, 1, 1), stream,
                *scan1_args)

        # }}}

        # {{{ second level scan of per-interval results

        # can scan at most one interval
        assert interval_size >= num_intervals

        scan2_args = data_args + [
                interval_results.gpudata,  # interval_sums
                ]
        if self.is_segmented:
            scan2_args.append(first_segment_start_in_interval.gpudata)
        scan2_args = scan2_args + [
                interval_results.gpudata,  # partial_scan_buffer
                num_intervals, interval_size]

        l2_evt = l2_info.kernel.prepared_async_call(
                (1, 1), (l1_info.wg_size, 1, 1), stream,
                *scan2_args)

        # }}}

        # {{{ update intervals with result of interval scan

        upd_args = data_args + [
                n, interval_size, interval_results.gpudata, partial_scan_buffer.gpudata]
        if self.is_segmented:
            upd_args.append(first_segment_start_in_interval.gpudata)
        if self.store_segment_start_flags:
            upd_args.append(segment_start_flags.gpudata)

        return self.final_update_info.kernel.prepared_async_call(
                (num_intervals, 1),
                (self.final_update_info.update_wg_size, 1, 1), stream,
                *upd_args)

        # }}}

# }}}


