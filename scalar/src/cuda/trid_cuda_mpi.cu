/*
 * Open source copyright declaration based on BSD open source template:
 * http://www.opensource.org/licenses/bsd-license.php
 *
 * This file is part of the scalar-tridiagonal solver distribution.
 *
 * Copyright (c) 2015, Endre László and others. Please see the AUTHORS file in
 * the main source directory for a full list of copyright holders.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * The name of Endre László may not be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY Endre László ''AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL Endre László BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

// Written by Endre Laszlo, University of Oxford, endre.laszlo@oerc.ox.ac.uk,
// 2013-2014

#include "trid_mpi_cuda.hpp"
#include <numeric>
#include <functional>

/*
 * Modified Thomas forwards pass in x direction
 * Each array should have a size of sys_size*sys_n, although the first element
 * of a (a[0]) in the first process and the last element of c in the last
 * process will not be used eventually
 */
template <typename REAL>
__global__ void
trid_linear_forward(const REAL *__restrict__ a, const REAL *__restrict__ b,
                    const REAL *__restrict__ c, const REAL *__restrict__ d,
                    const REAL *__restrict__ u, REAL *__restrict__ aa,
                    REAL *__restrict__ cc, REAL *__restrict__ dd,
                    REAL *__restrict__ boundaries, int sys_size, int sys_pads,
                    int sys_n) {

  REAL bb;
  int i;

  // Thread ID in global scope - every thread solves one system
  int tid = threadIdx.x + threadIdx.y * blockDim.x +
            blockIdx.x * blockDim.y * blockDim.x +
            blockIdx.y * gridDim.x * blockDim.y * blockDim.x;
  int ind = sys_pads * tid;

  if (tid < sys_n) {
    //
    // forward pass
    //
    for (i = 0; i < 2; ++i) {
      bb = static_cast<REAL>(1.0) / b[ind + i];
      dd[ind + i] = bb * d[ind + i];
      aa[ind + i] = bb * a[ind + i];
      cc[ind + i] = bb * c[ind + i];
    }

    if (sys_size >= 3) {
      // eliminate lower off-diagonal
      for (i = 2; i < sys_size; i++) {
        int loc_ind = ind + i;
        bb = static_cast<REAL>(1.0) /
             (b[loc_ind] - a[loc_ind] * cc[loc_ind - 1]);
        dd[loc_ind] = (d[loc_ind] - a[loc_ind] * dd[loc_ind - 1]) * bb;
        aa[loc_ind] = (-a[loc_ind] * aa[loc_ind - 1]) * bb;
        cc[loc_ind] = c[loc_ind] * bb;
      }
      // Eliminate upper off-diagonal
      for (i = sys_size - 3; i > 0; --i) {
        int loc_ind = ind + i;
        dd[loc_ind] = dd[loc_ind] - cc[loc_ind] * dd[loc_ind + 1];
        aa[loc_ind] = aa[loc_ind] - cc[loc_ind] * aa[loc_ind + 1];
        cc[loc_ind] = -cc[loc_ind] * cc[loc_ind + 1];
      }
      bb = static_cast<REAL>(1.0) /
           (static_cast<REAL>(1.0) - cc[ind] * aa[ind + 1]);
      dd[ind] = bb * (dd[ind] - cc[ind] * dd[ind + 1]);
      aa[ind] = bb * aa[ind];
      cc[ind] = bb * (-cc[ind] * cc[ind + 1]);
    }
    // prepare boundaries for communication
    i = tid * 6;
    boundaries[i + 0] = aa[ind];
    boundaries[i + 1] = aa[ind + sys_size - 1];
    boundaries[i + 2] = cc[ind];
    boundaries[i + 3] = cc[ind + sys_size - 1];
    boundaries[i + 4] = dd[ind];
    boundaries[i + 5] = dd[ind + sys_size - 1];
  }
}

//
// Modified Thomas backward pass
//
template <typename REAL>
__global__ void
trid_linear_backward(const REAL *__restrict__ aa, const REAL *__restrict__ cc,
                     const REAL *__restrict__ dd, REAL *__restrict__ d,
                     REAL *__restrict__ u, const REAL *__restrict__ boundaries,
                     int sys_size, int sys_pads, int sys_n) {
  // Thread ID in global scope - every thread solves one system
  int tid = threadIdx.x + threadIdx.y * blockDim.x +
            blockIdx.x * blockDim.y * blockDim.x +
            blockIdx.y * gridDim.x * blockDim.y * blockDim.x;
  int ind = sys_pads * tid;

  if (tid < sys_n) {
    //
    // reverse pass
    //
    REAL dd0 = boundaries[2 * tid], dd_last = boundaries[2 * tid + 1];
    d[ind] = dd0;
    for (int i = 1; i < sys_size - 1; i++) {
      d[ind + i] = dd[ind + i] - aa[ind + i] * dd0 - cc[ind + i] * dd_last;
    }
    d[ind + sys_size - 1] = dd_last;
  }
}


/*
 * Modified Thomas forward pass in y or higher dimensions.
 * Each array should have a size of sys_n*sys_size.
 * The layout and indexing of aa, cc, and dd are the same as of a, c, d
 * respectively
 * The boundaries array has a size of sys_n*6 and will hold the first and last
 * elements of aa, cc, and dd for each system
 *
 */
template <typename REAL>
__device__ void trid_strided_multidim_forward_kernel(
    const REAL *__restrict__ a, int ind_a, int stride_a,
    const REAL *__restrict__ b, int ind_b, int stride_b,
    const REAL *__restrict__ c, int ind_c, int stride_c,
    const REAL *__restrict__ d, int ind_d, int stride_d,
    const REAL *__restrict__ u, int ind_u, int stride_u, REAL *__restrict__ aa,
    REAL *__restrict__ cc, REAL *__restrict__ dd, REAL *__restrict__ boundaries,
    int ind_bound, int sys_size) {
  //
  // forward pass
  //
  REAL bb;
  for (int i = 0; i < 2; ++i) {
    bb = static_cast<REAL>(1.0) / b[ind_b + i * stride_b];
    cc[ind_c + i * stride_c] = bb * c[ind_c + i * stride_c];
    aa[ind_a + i * stride_a] = bb * a[ind_a + i * stride_a];
    dd[ind_d + i * stride_d] = bb * d[ind_d + i * stride_d];
  }

  if (sys_size >= 3) {
    // Eliminate lower off-diagonal
    for (int i = 2; i < sys_size; ++i) {
      bb = static_cast<REAL>(1.0) /
           (b[ind_b + i * stride_b] -
            a[ind_a + i * stride_a] * cc[ind_c + (i - 1) * stride_c]);
      dd[ind_d + i * stride_d] =
          (d[ind_d + i * stride_d] -
           a[ind_a + i * stride_a] * dd[ind_d + (i - 1) * stride_d]) *
          bb;
      aa[ind_a + i * stride_a] =
          (-a[ind_a + i * stride_a] * aa[ind_a + (i - 1) * stride_a]) * bb;
      cc[ind_c + i * stride_c] = c[ind_c + i * stride_c] * bb;
    }
    // Eliminate upper off-diagonal
    for (int i = sys_size - 3; i > 0; --i) {
      dd[ind_d + i * stride_d] =
          dd[ind_d + i * stride_d] -
          cc[ind_c + i * stride_c] * dd[ind_d + (i + 1) * stride_d];
      aa[ind_a + i * stride_a] =
          aa[ind_a + i * stride_a] -
          cc[ind_c + i * stride_c] * aa[ind_a + (i + 1) * stride_a];
      cc[ind_c + i * stride_c] =
          -cc[ind_c + i * stride_c] * cc[ind_c + (i + 1) * stride_c];
    }
    bb = static_cast<REAL>(1.0) /
         (static_cast<REAL>(1.0) - cc[ind_c] * aa[ind_a + stride_a]);
    dd[ind_d] = bb * (dd[ind_d] - cc[ind_c] * dd[ind_d + stride_d]);
    aa[ind_a] = bb * aa[ind_a];
    cc[ind_c] = bb * (-cc[ind_c] * cc[ind_c + stride_c]);
  }
  // prepare boundaries for communication
  boundaries[ind_bound + 0] = aa[ind_a];
  boundaries[ind_bound + 1] = aa[ind_a + (sys_size - 1) * stride_a];
  boundaries[ind_bound + 2] = cc[ind_c];
  boundaries[ind_bound + 3] = cc[ind_c + (sys_size - 1) * stride_c];
  boundaries[ind_bound + 4] = dd[ind_d];
  boundaries[ind_bound + 5] = dd[ind_d + (sys_size - 1) * stride_d];
}

template <typename REAL>
__global__ void trid_strided_multidim_forward(
    const REAL *__restrict__ a, const DIM_V a_pads, const REAL *__restrict__ b,
    const DIM_V b_pads, const REAL *__restrict__ c, const DIM_V c_pads,
    const REAL *__restrict__ d, const DIM_V d_pads, const REAL *__restrict__ u,
    const DIM_V u_pads, REAL *__restrict__ aa, REAL *__restrict__ cc,
    REAL *__restrict__ dd, REAL *__restrict__ boundaries, int ndim,
    int solvedim, int sys_n, const DIM_V dims) {
  // thread ID in block
  int tid = threadIdx.x + threadIdx.y * blockDim.x +
            threadIdx.z * blockDim.x * blockDim.y;
  if (solvedim < 1 || solvedim > ndim)
    return; /* Just hints to the compiler */

  int __shared__ d_cumdims[MAXDIM + 1];
  int __shared__ d_cumpads[5][MAXDIM + 1];

  /* Build up d_cumpads and d_cumdims */
  if (tid < 6) {
    int *tgt = (tid == 0) ? d_cumdims : d_cumpads[tid - 1];
    const int *src = NULL;
    switch (tid) {
    case 0:
      src = dims.v;
      break;
    case 1:
      src = a_pads.v;
      break;
    case 2:
      src = b_pads.v;
      break;
    case 3:
      src = c_pads.v;
      break;
    case 4:
      src = d_pads.v;
      break;
    case 5:
      src = u_pads.v;
      break;
    }

    tgt[0] = 1;
    for (int i = 0; i < ndim; i++) {
      tgt[i + 1] = tgt[i] * src[i];
    }
  }
  __syncthreads();
  //
  // set up indices for main block
  //
  // Thread ID in global scope - every thread solves one system
  tid = threadIdx.x + threadIdx.y * blockDim.x +
        blockIdx.x * blockDim.y * blockDim.x +
        blockIdx.y * gridDim.x * blockDim.y * blockDim.x;

  int ind_a = 0;
  int ind_b = 0;
  int ind_c = 0;
  int ind_d = 0;
  int ind_u = 0;
  int ind_bound = tid * 6;

  for (int j = 0; j < solvedim; j++) {
    ind_a += ((tid / d_cumdims[j]) % dims.v[j]) * d_cumpads[0][j];
    ind_b += ((tid / d_cumdims[j]) % dims.v[j]) * d_cumpads[1][j];
    ind_c += ((tid / d_cumdims[j]) % dims.v[j]) * d_cumpads[2][j];
    ind_d += ((tid / d_cumdims[j]) % dims.v[j]) * d_cumpads[3][j];
    // if (INC) ind_u += (( tid /  d_cumdims[j] ) % dims.v[j]) *
    // d_cumpads[4][j];
  }
  for (int j = solvedim + 1; j < ndim; j++) {
    ind_a += ((tid / (d_cumdims[j] / dims.v[solvedim])) % dims.v[j]) *
             d_cumpads[0][j];
    ind_b += ((tid / (d_cumdims[j] / dims.v[solvedim])) % dims.v[j]) *
             d_cumpads[1][j];
    ind_c += ((tid / (d_cumdims[j] / dims.v[solvedim])) % dims.v[j]) *
             d_cumpads[2][j];
    ind_d += ((tid / (d_cumdims[j] / dims.v[solvedim])) % dims.v[j]) *
             d_cumpads[3][j];
    // if (INC) ind_u += (( tid / (d_cumdims[j] / dims.v[solvedim])) %
    // dims.v[j]) * d_cumpads[4][j];
  }
  int stride_a = d_cumpads[0][solvedim];
  int stride_b = d_cumpads[1][solvedim];
  int stride_c = d_cumpads[2][solvedim];
  int stride_d = d_cumpads[3][solvedim];
  int stride_u = d_cumpads[4][solvedim];
  int sys_size = dims.v[solvedim];

  if (tid < sys_n) {
    trid_strided_multidim_forward_kernel<REAL>(
        a, ind_a, stride_a, b, ind_b, stride_b, c, ind_c, stride_c, d, ind_d,
        stride_d, u, ind_u, stride_u, aa, cc, dd, boundaries, ind_bound,
        sys_size);
  }
}


/*
 * Modified Thomas backward pass in y or higher dimensions.
 * Each array should have a size of sys_n*sys_size.
 * The layout and indexing of aa, cc, and dd are the same as of a, c, d
 * respectively
 * The boundaries array has a size of sys_n*2 and hold the first and last
 * elements of dd for each system
 *
 */
template <typename REAL>
__device__ void trid_strided_multidim_backward_kernel(
    const REAL *__restrict__ aa, int ind_a, int stride_a,
    const REAL *__restrict__ cc, int ind_c, int stride_c,
    const REAL *__restrict__ dd, REAL *__restrict__ d, int ind_d, int stride_d,
    REAL *__restrict__ u, int ind_u, int stride_u,
    const REAL *__restrict__ boundaries, int ind_bound, int sys_size) {
  //
  // reverse pass
  //
  REAL dd0 = boundaries[ind_bound], dd_last = boundaries[ind_bound + 1];
  d[ind_d] = dd0;
  for (int i = 1; i < sys_size - 1; i++) {
    d[ind_d + i * stride_d] = dd[ind_d + i * stride_d] -
                              aa[ind_a + i * stride_a] * dd0 -
                              cc[ind_c + i * stride_c] * dd_last;
  }
  d[ind_d + (sys_size - 1) * stride_d] = dd_last;
}

template <typename REAL>
__global__ void
trid_strided_multidim_backward(const REAL *__restrict__ aa, const DIM_V a_pads,
                               const REAL *__restrict__ cc, const DIM_V c_pads,
                               const REAL *__restrict__ dd,
                               REAL *__restrict__ d, const DIM_V d_pads,
                               REAL *__restrict__ u, const DIM_V u_pads,
                               const REAL *__restrict__ boundaries, int ndim,
                               int solvedim, int sys_n, const DIM_V dims) {
  // thread ID in block
  int tid = threadIdx.x + threadIdx.y * blockDim.x +
            threadIdx.z * blockDim.x * blockDim.y;
  if (solvedim < 1 || solvedim > ndim)
    return; /* Just hints to the compiler */

  int __shared__ d_cumdims[MAXDIM + 1];
  int __shared__ d_cumpads[5][MAXDIM + 1];

  /* Build up d_cumpads and d_cumdims */
  if (tid < 5) {
    int *tgt = (tid == 0) ? d_cumdims : d_cumpads[tid - 1];
    const int *src = NULL;
    switch (tid) {
    case 0:
      src = dims.v;
      break;
    case 1:
      src = a_pads.v;
      break;
    case 2:
      src = c_pads.v;
      break;
    case 3:
      src = d_pads.v;
      break;
    case 4:
      src = u_pads.v;
      break;
    }

    tgt[0] = 1;
    for (int i = 0; i < ndim; i++) {
      tgt[i + 1] = tgt[i] * src[i];
    }
  }
  __syncthreads();
  //
  // set up indices for main block
  //
  // Thread ID in global scope - every thread solves one system
  tid = threadIdx.x + threadIdx.y * blockDim.x +
        blockIdx.x * blockDim.y * blockDim.x +
        blockIdx.y * gridDim.x * blockDim.y * blockDim.x;

  int ind_a = 0;
  int ind_c = 0;
  int ind_d = 0;
  int ind_u = 0;
  int ind_bound = tid * 2; // 2 values per system since it hold only dd

  for (int j = 0; j < solvedim; j++) {
    ind_a += ((tid / d_cumdims[j]) % dims.v[j]) * d_cumpads[0][j];
    ind_c += ((tid / d_cumdims[j]) % dims.v[j]) * d_cumpads[2][j];
    ind_d += ((tid / d_cumdims[j]) % dims.v[j]) * d_cumpads[3][j];
    // if (INC) ind_u += (( tid /  d_cumdims[j] ) % dims.v[j]) *
    // d_cumpads[4][j];
  }
  for (int j = solvedim + 1; j < ndim; j++) {
    ind_a += ((tid / (d_cumdims[j] / dims.v[solvedim])) % dims.v[j]) *
             d_cumpads[0][j];
    ind_c += ((tid / (d_cumdims[j] / dims.v[solvedim])) % dims.v[j]) *
             d_cumpads[2][j];
    ind_d += ((tid / (d_cumdims[j] / dims.v[solvedim])) % dims.v[j]) *
             d_cumpads[3][j];
    // if (INC) ind_u += (( tid / (d_cumdims[j] / dims.v[solvedim])) %
    // dims.v[j]) * d_cumpads[4][j];
  }
  int stride_a = d_cumpads[0][solvedim];
  int stride_c = d_cumpads[2][solvedim];
  int stride_d = d_cumpads[3][solvedim];
  int stride_u = d_cumpads[4][solvedim];
  int sys_size = dims.v[solvedim];

  if (tid < sys_n) {
    trid_strided_multidim_backward_kernel<REAL>(
        aa, ind_a, stride_a, cc, ind_c, stride_c, dd, d, ind_d, stride_d, u,
        ind_u, stride_u, boundaries, ind_bound, sys_size);
  }
}
//
// Thomas solver for reduced systems
//
// Solve sys_n reduced systems from receive_buf (the result of the allgather
// with boundaries of mpi nodes) and stores dd[0] and dd[local_size-1] of the
// current node for each system in results.
// num_proc: number of processes along the solving dimension
// mpi_coord: index of the current process along the solving dimension
//
template <typename REAL>
void thomas_on_reduced_batched(const REAL *receive_buf, REAL *results,
                               int sys_n, int num_proc, int mpi_coord) {
#pragma omp parallel for
  for (size_t eq_idx = 0; eq_idx < sys_n; ++eq_idx) {
    // Reduced system
    std::vector<REAL> aa_r(2 * num_proc),
        cc_r(2 * num_proc),
        dd_r(2 * num_proc);
    // The offset in the send and receive buffers
    const size_t comm_buf_offset = eq_idx * 6;
    const size_t comm_buf_size = 6 * sys_n;
    for (int i = 0; i < num_proc; ++i) {
      aa_r[2 * i + 0] = receive_buf[comm_buf_size * i + comm_buf_offset + 0];
      aa_r[2 * i + 1] = receive_buf[comm_buf_size * i + comm_buf_offset + 1];
      cc_r[2 * i + 0] = receive_buf[comm_buf_size * i + comm_buf_offset + 2];
      cc_r[2 * i + 1] = receive_buf[comm_buf_size * i + comm_buf_offset + 3];
      dd_r[2 * i + 0] = receive_buf[comm_buf_size * i + comm_buf_offset + 4];
      dd_r[2 * i + 1] = receive_buf[comm_buf_size * i + comm_buf_offset + 5];
    }

    // indexing of cc_r, dd_r starts from 0
    // while indexing of aa_r starts from 1
    thomas_on_reduced<REAL>(aa_r.data(), cc_r.data(), dd_r.data(),
                            2 * num_proc, 1);

    results[2 * eq_idx + 0] = dd_r[2 * mpi_coord + 0];
    results[2 * eq_idx + 1] = dd_r[2 * mpi_coord + 1];
  }
}

template <typename REAL>
void tridMultiDimBatchSolveMPI(const MpiSolverParams &params, const REAL *a,
                               int *a_pads, const REAL *b, int *b_pads,
                               const REAL *c, int *c_pads, REAL *d, int *d_pads,
                               REAL *u, int *u_pads) {
  // TODO paddings!!
  assert(params.equation_dim < params.num_dims);
  assert((
      (std::is_same<REAL, float>::value || std::is_same<REAL, double>::value) &&
      "trid_solve_mpi: only double or float values are supported"));

  // The size of the equations / our domain
  const size_t local_eq_size = params.size[params.equation_dim];
  assert(local_eq_size > 2 &&
         "One of the processes has fewer than 2 equations, this is not "
         "supported\n");
  const int eq_stride =
      std::accumulate(params.size, params.size + params.equation_dim, 1,
                      std::multiplies<int>{});

  // The product of the sizes along the dimensions higher than solve_dim; needed
  // for the iteration later
  const int outer_size =
      std::accumulate(params.size + params.equation_dim + 1,
                      params.size + params.num_dims, 1, std::multiplies<int>{});

  // The number of systems to solve
  const int sys_n = eq_stride * outer_size;

  const MPI_Datatype real_datatype =
      std::is_same<REAL, double>::value ? MPI_DOUBLE : MPI_FLOAT;

  const int local_helper_size = outer_size * eq_stride * local_eq_size;
  REAL *aa, *cc, *dd, *boundaries;
  cudaMalloc(&aa, local_helper_size * sizeof(REAL));
  cudaMalloc(&cc, local_helper_size * sizeof(REAL));
  cudaMalloc(&dd, local_helper_size * sizeof(REAL));
  cudaMalloc(&boundaries, sys_n * 6 * sizeof(REAL));

  int blockdimx = 128; // Has to be the multiple of 4(or maybe 32??)
  int blockdimy = 1;
  int dimgrid = 1 + (sys_n - 1) / blockdimx; // can go up to 65535
  int dimgridx = dimgrid % 65536;            // can go up to max 65535 on Fermi
  int dimgridy = 1 + dimgrid / 65536;

  dim3 dimGrid_x(dimgridx, dimgridy);
  dim3 dimBlock_x(blockdimx, blockdimy);
  if (params.equation_dim == 0) {
    trid_linear_forward<REAL>
        <<<dimGrid_x, dimBlock_x>>>(a, b, c, d, u, aa, cc, dd, boundaries,
                                    local_eq_size, local_eq_size, sys_n);
  } else {
    DIM_V pads, dims;
    for (int i = 0; i < params.num_dims; ++i) {
      pads.v[i] = a_pads[i];
      dims.v[i] = a_pads[i];
    }
    trid_strided_multidim_forward<REAL><<<dimGrid_x, dimBlock_x>>>(
        a, pads, b, pads, c, pads, d, pads, u, pads, aa, cc, dd,
        boundaries, params.num_dims, params.equation_dim, sys_n, dims);
  }
  // MPI buffers (6 because 2 from each of the a, c and d coefficient arrays)
  const size_t comm_buf_size = 6 * sys_n;
  std::vector<REAL> send_buf(comm_buf_size),
      receive_buf(comm_buf_size * params.num_mpi_procs[params.equation_dim]);
  cudaMemcpy(send_buf.data(), boundaries, sizeof(REAL) * comm_buf_size,
             cudaMemcpyDeviceToHost);
  // Communicate boundary results
  MPI_Allgather(send_buf.data(), comm_buf_size, real_datatype,
                receive_buf.data(), comm_buf_size, real_datatype,
                params.communicator);
  // solve reduced systems, and store results in send_buf
  thomas_on_reduced_batched<REAL>(receive_buf.data(), send_buf.data(), sys_n,
                                  params.num_mpi_procs[params.equation_dim],
                                  params.mpi_coord);

  // copy the results of the reduced systems to the beginning of the boundaries
  // array
  cudaMemcpy(boundaries, send_buf.data(), sizeof(REAL) * 2 * sys_n,
             cudaMemcpyHostToDevice);

  if (params.equation_dim == 0) {
    trid_linear_backward<REAL><<<dimGrid_x, dimBlock_x>>>(
        aa, cc, dd, d, u, boundaries, local_eq_size, local_eq_size, sys_n);
  } else {
    DIM_V pads, dims;
    for (int i = 0; i < params.num_dims; ++i) {
      pads.v[i] = a_pads[i];
      dims.v[i] = a_pads[i];
    }
    trid_strided_multidim_backward<REAL><<<dimGrid_x, dimBlock_x>>>(
        aa, pads, cc, pads, dd, d, pads, u, pads, boundaries, params.num_dims,
        params.equation_dim, sys_n, dims);
  }

  cudaFree(aa);
  cudaFree(cc);
  cudaFree(dd);
  cudaFree(boundaries);
}

template <typename REAL>
void tridMultiDimBatchSolveMPI(const MpiSolverParams &params, const REAL *a,
                               const REAL *b, const REAL *c, REAL *d, REAL *u,
                               int *pads) {
  tridMultiDimBatchSolveMPI<REAL>(params, a, pads, b, pads, c, pads, d, pads, u, pads);
}


EXTERN_C
tridStatus_t tridDmtsvStridedBatchMPI(const MpiSolverParams &params,
                                      const double *a, const double *b,
                                      const double *c, double *d, double *u,
                                      int *pads) {
  tridMultiDimBatchSolveMPI<double>(params, a, b, c, d, u, pads);
  return TRID_STATUS_SUCCESS;
}

EXTERN_C
tridStatus_t tridSmtsvStridedBatchMPI(const MpiSolverParams &params,
                                      const float *a, const float *b,
                                      const float *c, float *d, float *u,
                                      int *pads){
  tridMultiDimBatchSolveMPI<float>(params, a, b, c, d, u, pads);
  return TRID_STATUS_SUCCESS;
}

EXTERN_C
tridStatus_t tridDmtsvStridedBatchPaddedMPI(const MpiSolverParams &params,
                                            const double *a, int *a_pads,
                                            const double *b, int *b_pads,
                                            const double *c, int *c_pads,
                                            double *d, int *d_pads, double *u,
                                            int *u_pads) {
  tridMultiDimBatchSolveMPI<double>(params, a, a_pads, b, b_pads, c, c_pads, d,
                                    d_pads, u, u_pads);
  return TRID_STATUS_SUCCESS;
}

EXTERN_C
tridStatus_t tridSmtsvStridedBatchPaddedMPI(const MpiSolverParams &params,
                                            const float *a, int *a_pads,
                                            const float *b, int *b_pads,
                                            const float *c, int *c_pads,
                                            float *d, int *d_pads, float *u,
                                            int *u_pads) {
  tridMultiDimBatchSolveMPI<float>(params, a, a_pads, b, b_pads, c, c_pads, d,
                                   d_pads, u, u_pads);
  return TRID_STATUS_SUCCESS;
}


