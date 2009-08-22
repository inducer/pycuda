#raw
#include <stdio.h>
#end raw

// =============================================================================
// PyCuda Optimized Matrix Multiplication 
// Template Meta-programming Example using Cheetah 
// -----------------------------------------------------------------------------
// These are the five performance-tuning parameters at your disposal
// -----------------------------------------------------------------------------

// Thread block size 
// (available values are 1, 2, 4, 8, 12 and 16)
#if $BLOCK_SIZE not in [1, 2, 4, 8, 12, 16]
#raise ValueError, "$BLOCK_SIZE not in [1, 2, 4, 8, 12, 16]"
#end if

// Work size, or number of matrix N tiles per thread 
// (available values are 1, 2 and 4)
#if $WORK_SIZE not in [1, 2, 4]
#raise ValueError, "$WORK_SIZE not in [1, 2, 4]"
#end if

// Dot product loop unrolling factor
// (available values are 0, 1, 3, 7 and 15)
#if $UNROLL not in [0, 1, 3, 7, 15]
#raise ValueError, "$UNROLL not in [0, 1, 3, 7, 15]"
#end if

// Register spilling
// (boolean)
#if type($SPILL) != bool
#raise ValueError, "type($SPILL) != bool"
#end if

// Prefetching 
// (boolean)
#if type($PREFETCH) != bool
#raise ValueError, "type($PREFETCH) != bool"
#end if

// =============================================================================
__global__ void
matrixMul(float* C, float* A, float* B)
{
  // Block index
  const uint bx = blockIdx.x;
  const uint by = blockIdx.y;
  
  // Thread index
  const uint tx = threadIdx.x;
  const uint ty = threadIdx.y;
  
  // Index of the first sub-matrix of A processed by this block
  const uint aBegin = $A_WIDTH * $BLOCK_SIZE * by;
  // Index of the last element of the sub-matrix of A processed by this block
  const uint aEnd = aBegin + $A_WIDTH * (ty + 1) + tx;
  // Step size used to iterate through the sub-matrices of A
  const uint aStep = $BLOCK_SIZE;

  // Index of the first sub-matrix of B processed by this block
  const uint bBegin = $BLOCK_SIZE * $WORK_SIZE * bx;  
  // Step size used to iterate through the sub-matrices of B
  const uint bStep = $BLOCK_SIZE * $B_WIDTH;

  // Index of the output value for this thread
#if $SPILL
  // Create a shared-memory buffer to spill a register value
  // into shared memory, hopefully reducing the total required
  // register count.
  __shared__ int c[$BLOCK_SIZE][$BLOCK_SIZE];
  c[tx][ty] = bBegin + $B_WIDTH * $BLOCK_SIZE * by + $B_WIDTH * ty + tx;
#else
  const uint c =  bBegin + $B_WIDTH * $BLOCK_SIZE * by + $B_WIDTH * ty + tx;
#end if // $SPILL

  // Initialize (sub)result(s) to 0.
  float sub[$WORK_SIZE];
#for w in xrange($WORK_SIZE)
  sub[$w] = 0;
#end for

  // Current indexes
  uint a = aBegin + $A_WIDTH * ty + tx;
  uint b = bBegin + $B_WIDTH * ty + tx;

#if not $PREFETCH
  // ---------------------------------------------------------------------------
  // Code *without* prefetching
  // ---------------------------------------------------------------------------  
  // Loop over all the sub-matrices of A and B
  // required to compute the block sub-matrix
  while (a < aEnd) {

    // Shared memory for the sub-matrix of A
    __shared__ float As[$BLOCK_SIZE][$BLOCK_SIZE];
    // Shared memory for the sub-matrix of B
    __shared__ float Bs[$BLOCK_SIZE][$BLOCK_SIZE * $WORK_SIZE];

    // Load the matrices from device memory
    // directly to shared memory
    As[ty][tx] = A[a];
#for w in xrange($WORK_SIZE)
    Bs[ty][tx + $BLOCK_SIZE * $w] = B[b + $BLOCK_SIZE * $w];
#end for // w in xrange($WORK_SIZE)
    
    // Update for next loop
    a += aStep;
    b += bStep;

    // Synchronize to make sure the shared memory
    // tiles are ready
    __syncthreads();

    // Compute dot-product (with easy unroll ;-)
    for (int i = 0; i < $BLOCK_SIZE; i += $UNROLL + 1)
      {
#for u in xrange(min($BLOCK_SIZE, $UNROLL + 1))
#for w in xrange($WORK_SIZE)
	sub[$w] += As[ty][i+$u] * Bs[i+$u][tx + $BLOCK_SIZE * $w];
#end for // w in xrange($WORK_SIZE)
#end for // u in xrange($UNROLL + 1)
      }

    // Synchronize to make sure that the preceding
    // computation is done before overwriting new
    // shared memory sub-matrices of A and B in the next iteration
    __syncthreads();
  }

#else

  // ---------------------------------------------------------------------------
  // Code *with* prefetching
  // ---------------------------------------------------------------------------  
  // Initial prefetch.  Issues loads to main memory and store 
  // in temporary variables which will later be stored to shared memory
  float fa = A[a];
  float fb[$WORK_SIZE];
#for w in xrange($WORK_SIZE)
  fb[$w] = B[b + $BLOCK_SIZE * $w];
#end for // w in xrange($WORK_SIZE)
  
  // Shared memory for the sub-matrix of A
  __shared__ float As[$BLOCK_SIZE][$BLOCK_SIZE];
  // Shared memory for the sub-matrix of B
  __shared__ float Bs[$BLOCK_SIZE][$BLOCK_SIZE * $WORK_SIZE];

  // ---------------------------------------------------------------------------
  // Loop over all the sub-matrices of A and B
  // required to compute the block sub-matrix
  while (a < (aEnd-aStep)) {

    // When performing prefetching, the values are already loaded
    // from memory, and the temporary variables holding the loaded
    // values are stored to shared memory.
    As[ty][tx] = fa;
#for w in xrange($WORK_SIZE)
    Bs[ty][tx + $BLOCK_SIZE * $w] = fb[$w];
#end for // w in xrange($WORK_SIZE)
    
    // Update for next loop
    a += aStep;
    b += bStep;

    // Synchronize to make sure the shared memory
    // tiles are ready
    __syncthreads();

    // Issue the loads for the next tiles preemptively.
    // The loads will complete and be stored into these temporary
    // variables while the current shared memory tiles
    // are being operated on.
    fa = A[a];
#for w in xrange($WORK_SIZE)
    fb[$w] = B[b + $BLOCK_SIZE * $w];
#end for // w in xrange($WORK_SIZE)

    // Compute dot-product (with easy unroll ;-)
    for (int i = 0; i < $BLOCK_SIZE; i += $UNROLL + 1)
      {
#for u in xrange(min($BLOCK_SIZE, $UNROLL + 1))
#for w in xrange($WORK_SIZE)
	sub[$w] += As[ty][i+$u] * Bs[i+$u][tx + $BLOCK_SIZE * $w];
#end for // w in xrange($WORK_SIZE)
#end for // u in xrange($UNROLL + 1)
      }

    // Synchronize to make sure that the preceding
    // computation is done before overwriting new
    // shared memory sub-matrices of A and B in the next iteration
    __syncthreads();
  }

  // Last iteration (with no pre-emptive loading) 
  As[ty][tx] = fa;
#for w in xrange($WORK_SIZE)
  Bs[ty][tx + $BLOCK_SIZE * $w] = fb[$w];
#end for // w in xrange($WORK_SIZE)
    
  // Update for next loop
  a += aStep;
  b += bStep;

  // Synchronize to make sure the shared memory
  // tiles are ready
  __syncthreads();

  // Compute dot-product (with easy unroll ;-)
  for (int i = 0; i < $BLOCK_SIZE; i += $UNROLL + 1)
    {
#for u in xrange(min($BLOCK_SIZE, $UNROLL + 1))
#for w in xrange($WORK_SIZE)
      sub[$w] += As[ty][i+$u] * Bs[i+$u][tx + $BLOCK_SIZE * $w];
#end for // w in xrange($WORK_SIZE)
#end for // u in xrange($UNROLL + 1)
    }

  // Synchronize to make sure that the preceding
  // computation is done before overwriting new
  // shared memory sub-matrices of A and B in the next iteration
  __syncthreads();

#end if // not $PRETETCH 

  // ---------------------------------------------------------------------------
  // Output the final result(s) for each thread.
#for w in xrange($WORK_SIZE)
#if $SPILL
  // If we spilled the output index at the beginning, load it back
  // from the shared memory array.
  C[c[tx][ty] + $BLOCK_SIZE * $w] = sub[$w];
#else
  C[c + $BLOCK_SIZE * $w] = sub[$w];
#end if // $SPILL
#end for // w in xrange($WORK_SIZE)

}
