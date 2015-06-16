from __future__ import absolute_import
import numpy as np
import pycuda.gpuarray as gpuarray
from six.moves import range




def build_pkt_data_structure(spmv, packet_nr_to_dofs, max_thread_costs,
        old2new_fetch_indices, csr_mat, thread_count, thread_assignments,
        local_row_costs):
    packet_start = 0
    base_dof_nr = 0

    index_array = np.zeros(
            max_thread_costs*thread_count, dtype=spmv.packed_index_dtype)
    data_array = np.zeros(
            max_thread_costs*thread_count, dtype=spmv.dtype)
    thread_starts = np.zeros(
            thread_count, dtype=spmv.index_dtype)
    thread_ends = np.zeros(
            thread_count, dtype=spmv.index_dtype)

    for packet_nr, packet_dofs in enumerate(packet_nr_to_dofs):
        base_thread_nr = packet_nr*spmv.threads_per_packet
        max_packet_items = 0

        for thread_offset in range(spmv.threads_per_packet):
            thread_write_idx = packet_start+thread_offset
            thread_start = packet_start+thread_offset
            thread_starts[base_thread_nr+thread_offset] = thread_write_idx

            for row_nr in thread_assignments[base_thread_nr+thread_offset]:
                perm_row_nr = old2new_fetch_indices[row_nr]
                rel_row_nr = perm_row_nr - base_dof_nr
                assert 0 <= rel_row_nr < len(packet_dofs)

                row_entries = 0

                for idx in range(csr_mat.indptr[row_nr], csr_mat.indptr[row_nr+1]):
                    col_nr = csr_mat.indices[idx]

                    perm_col_nr = old2new_fetch_indices[col_nr]
                    rel_col_nr = perm_col_nr - base_dof_nr

                    if 0 <= rel_col_nr < len(packet_dofs):
                        index_array[thread_write_idx] = (rel_row_nr << 16) + rel_col_nr
                        data_array[thread_write_idx] = csr_mat.data[idx]
                        thread_write_idx += spmv.threads_per_packet
                        row_entries += 1

                assert row_entries == local_row_costs[row_nr]

            thread_ends[base_thread_nr+thread_offset] = thread_write_idx

            thread_items = (thread_write_idx - thread_start)//spmv.threads_per_packet
            max_packet_items = max(
                    max_packet_items, thread_items)

        base_dof_nr += len(packet_dofs)
        packet_start += max_packet_items*spmv.threads_per_packet

    spmv.thread_starts = gpuarray.to_gpu(thread_starts)
    spmv.thread_ends = gpuarray.to_gpu(thread_ends)
    spmv.index_array = gpuarray.to_gpu(index_array)
    spmv.data_array = gpuarray.to_gpu(data_array)




try:
    import pyximport
except ImportError:
    pass
else:
    pyximport.install()
    from pycuda.sparse.pkt_build_cython import build_pkt_data_structure
