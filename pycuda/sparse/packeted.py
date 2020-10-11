from __future__ import division
from __future__ import absolute_import
from __future__ import print_function
from pytools import memoize_method
import pycuda.gpuarray as gpuarray
from pycuda.compiler import SourceModule
import numpy as np


PKT_KERNEL_TEMPLATE = """
typedef %(index_type)s index_type;
typedef %(value_type)s value_type;
typedef %(packed_index_type)s packed_index_type;

#define ROWS_PER_PACKET %(rows_per_packet)d
#define THREADS_PER_PACKET %(threads_per_packet)d

template <typename IndexType, typename ValueType>
__device__ void memcpy_device(
  ValueType *dest, const ValueType *src,
  const IndexType num_values)
{
  for(unsigned int i = threadIdx.x; i < num_values; i += blockDim.x)
  {
    dest[i] = src[i];
  }
}

#define pkt_unpack_row_index(packed_index) ( packed_index >> 16  )
#define pkt_unpack_col_index(packed_index) (packed_index & 0xFFFF)

extern "C" {
__global__ void
spmv_pkt_kernel(const index_type *row_ptr,
                const index_type *pos_start,
                const index_type *pos_end,
                const packed_index_type *index_array,
                const value_type *data_array,
                const value_type *x,
                      value_type *y)
{
  __shared__ value_type s_x[ROWS_PER_PACKET]; // input x-values
  __shared__ value_type s_y[ROWS_PER_PACKET]; // output y-values

  const index_type thread_id =
    __umul24(THREADS_PER_PACKET, blockIdx.x) + threadIdx.x;

  // base index of the submatrix corresponding to this packet
  const index_type packet_base_row = row_ptr[blockIdx.x];
  const index_type packet_num_rows = row_ptr[blockIdx.x+1] - packet_base_row;

  // copy local x and y values from global memory into shared memory
  memcpy_device(s_x, x + packet_base_row, packet_num_rows);
  memcpy_device(s_y, y + packet_base_row, packet_num_rows);

  __syncthreads();

  // process packet

  const index_type packet_start = pos_start[thread_id];
  const index_type packet_end = pos_end[thread_id];

  for(index_type pos = packet_start; pos != packet_end; pos += THREADS_PER_PACKET)
  {
    // row and column indices are stored in the same 32-bit word

    const index_type packed_index = index_array[pos];

    const index_type row = pkt_unpack_row_index(packed_index);
    const index_type col = pkt_unpack_col_index(packed_index);
    const value_type val = data_array[pos];

    s_y[row] += val * s_x[col];
  }

  __syncthreads();

  // copy y-values from shared memory to global array

  memcpy_device(y + packet_base_row, s_y, packet_num_rows);
}
}
"""


class PacketedSpMV:
    def __init__(self, mat, is_symmetric, dtype):
        from pycuda.tools import DeviceData

        devdata = DeviceData()

        # all row indices in the data structure generation code are
        # "unpermuted" unless otherwise specified
        self.dtype = np.dtype(dtype)
        self.index_dtype = np.int32
        self.packed_index_dtype = np.uint32
        self.threads_per_packet = devdata.max_threads

        h, w = self.shape = mat.shape
        if h != w:
            raise ValueError("only square matrices are supported")

        self.rows_per_packet = (devdata.shared_memory - 100) // (
            2 * self.dtype.itemsize
        )

        self.block_count = (h + self.rows_per_packet - 1) // self.rows_per_packet

        # get metis partition -------------------------------------------------
        from scipy.sparse import csr_matrix

        csr_mat = csr_matrix(mat, dtype=self.dtype)

        from pymetis import part_graph

        if not is_symmetric:
            # make sure adjacency graph is undirected
            adj_mat = csr_mat + csr_mat.T
        else:
            adj_mat = csr_mat

        while True:
            cut_count, dof_to_packet_nr = part_graph(
                int(self.block_count), xadj=adj_mat.indptr, adjncy=adj_mat.indices
            )

            # build packet_nr_to_dofs
            packet_nr_to_dofs = {}
            for i, packet_nr in enumerate(dof_to_packet_nr):
                try:
                    dof_packet = packet_nr_to_dofs[packet_nr]
                except KeyError:
                    packet_nr_to_dofs[packet_nr] = dof_packet = []

                dof_packet.append(i)

            packet_nr_to_dofs = [
                packet_nr_to_dofs.get(i) for i in range(len(packet_nr_to_dofs))
            ]

            too_big = False
            for packet_dofs in packet_nr_to_dofs:
                if len(packet_dofs) >= self.rows_per_packet:
                    too_big = True
                    break

            if too_big:
                old_block_count = self.block_count
                self.block_count = int(2 + 1.05 * self.block_count)
                print(
                    (
                        "Metis produced a big block at block count "
                        "%d--retrying with %d" % (old_block_count, self.block_count)
                    )
                )
                continue

            break

        assert len(packet_nr_to_dofs) == self.block_count

        # permutations, base rows ---------------------------------------------
        (
            new2old_fetch_indices,
            old2new_fetch_indices,
            packet_base_rows,
        ) = self.find_simple_index_stuff(packet_nr_to_dofs)

        # find local row cost and remaining_coo -------------------------------
        local_row_costs, remaining_coo = self.find_local_row_costs_and_remaining_coo(
            csr_mat, dof_to_packet_nr, old2new_fetch_indices
        )
        local_nnz = np.sum(local_row_costs)

        assert remaining_coo.nnz == csr_mat.nnz - local_nnz

        # find thread assignment for each block -------------------------------
        thread_count = len(packet_nr_to_dofs) * self.threads_per_packet
        thread_assignments, thread_costs = self.find_thread_assignment(
            packet_nr_to_dofs, local_row_costs, thread_count
        )

        max_thread_costs = np.max(thread_costs)

        # build data structure ------------------------------------------------
        from .pkt_build import build_pkt_data_structure

        build_pkt_data_structure(
            self,
            packet_nr_to_dofs,
            max_thread_costs,
            old2new_fetch_indices,
            csr_mat,
            thread_count,
            thread_assignments,
            local_row_costs,
        )

        self.packet_base_rows = gpuarray.to_gpu(packet_base_rows)
        self.new2old_fetch_indices = gpuarray.to_gpu(new2old_fetch_indices)
        self.old2new_fetch_indices = gpuarray.to_gpu(old2new_fetch_indices)

        from .coordinate import CoordinateSpMV

        self.remaining_coo_gpu = CoordinateSpMV(remaining_coo, dtype)

    def find_simple_index_stuff(self, packet_nr_to_dofs):
        new2old_fetch_indices = np.zeros(self.shape[0], dtype=self.index_dtype)
        old2new_fetch_indices = np.zeros(self.shape[0], dtype=self.index_dtype)

        packet_base_rows = np.zeros(self.block_count + 1, dtype=self.index_dtype)

        row_start = 0
        for packet_nr, packet in enumerate(packet_nr_to_dofs):
            packet_base_rows[packet_nr] = row_start
            row_end = row_start + len(packet)

            pkt_indices = np.array(packet, dtype=self.index_dtype)
            new2old_fetch_indices[row_start:row_end] = pkt_indices
            old2new_fetch_indices[pkt_indices] = np.arange(
                row_start, row_end, dtype=self.index_dtype
            )

            row_start += len(packet)

        packet_base_rows[self.block_count] = row_start

        return (new2old_fetch_indices, old2new_fetch_indices, packet_base_rows)

    def find_local_row_costs_and_remaining_coo(
        self, csr_mat, dof_to_packet_nr, old2new_fetch_indices
    ):
        h, w = self.shape
        local_row_costs = [0] * h
        rem_coo_values = []
        rem_coo_i = []
        rem_coo_j = []

        iptr = csr_mat.indptr
        indices = csr_mat.indices
        data = csr_mat.data

        for i in range(h):
            for idx in range(iptr[i], iptr[i + 1]):
                j = indices[idx]

                if dof_to_packet_nr[i] == dof_to_packet_nr[j]:
                    local_row_costs[i] += 1
                else:
                    rem_coo_values.append(data[idx])
                    rem_coo_i.append(old2new_fetch_indices[i])
                    rem_coo_j.append(old2new_fetch_indices[j])

        from scipy.sparse import coo_matrix

        remaining_coo = coo_matrix(
            (rem_coo_values, (rem_coo_i, rem_coo_j)), self.shape, dtype=self.dtype
        )

        return local_row_costs, remaining_coo

    def find_thread_assignment(self, packet_nr_to_dofs, local_row_cost, thread_count):
        thread_assignments = [[] for i in range(thread_count)]
        thread_costs = np.zeros(thread_count)

        for packet_nr, packet_dofs in enumerate(packet_nr_to_dofs):
            row_costs_and_numbers = sorted(
                [(local_row_cost[i], i) for i in packet_dofs], reverse=True
            )

            base_thread_nr = packet_nr * self.threads_per_packet
            thread_offset = 0

            # zigzag assignment
            step = 1
            for row_cost, row_number in row_costs_and_numbers:
                ti = base_thread_nr + thread_offset
                thread_assignments[ti].append(row_number)
                thread_costs[ti] += row_cost

                if thread_offset + step >= self.threads_per_packet:
                    step = -1
                elif thread_offset + step < 0:
                    step = 1
                else:
                    thread_offset += step

        return thread_assignments, thread_costs

    def build_gpu_data_structure(
        self,
        packet_nr_to_dofs,
        max_thread_costs,
        old2new_fetch_indices,
        csr_mat,
        thread_count,
        thread_assignments,
        local_row_costs,
    ):
        # these arrays will likely be too long, but that's ok

        from .pkt_build import build_pkt_structure

        build_pkt_structure(
            self,
            packet_nr_to_dofs,
            thread_assignments,
            # thread_starts,
            # thread_ends,
            # index_array,
            # data_array,
        )

        # copy data to the gpu ------------------------------------------------

    # execution ---------------------------------------------------------------
    @memoize_method
    def get_kernel(self):
        from pycuda.tools import dtype_to_ctype

        mod = SourceModule(
            PKT_KERNEL_TEMPLATE
            % {
                "value_type": dtype_to_ctype(self.dtype),
                "index_type": dtype_to_ctype(self.index_dtype),
                "packed_index_type": dtype_to_ctype(self.packed_index_dtype),
                "threads_per_packet": self.threads_per_packet,
                "rows_per_packet": self.rows_per_packet,
            },
            no_extern_c=True,
        )
        func = mod.get_function("spmv_pkt_kernel")
        func.prepare("PPPPPPP")
        return func

    def permute(self, x):
        return gpuarray.take(x, self.new2old_fetch_indices)

    def unpermute(self, x):
        return gpuarray.take(x, self.old2new_fetch_indices)

    def __call__(self, x, y=None):
        if y is None:
            y = gpuarray.zeros(self.shape[0], dtype=self.dtype, allocator=x.allocator)

        self.get_kernel().prepared_call(
            (self.block_count, 1),
            (self.threads_per_packet, 1, 1),
            self.packet_base_rows.gpudata,
            self.thread_starts.gpudata,
            self.thread_ends.gpudata,
            self.index_array.gpudata,
            self.data_array.gpudata,
            x.gpudata,
            y.gpudata,
        )

        self.remaining_coo_gpu(x, y)

        return y
