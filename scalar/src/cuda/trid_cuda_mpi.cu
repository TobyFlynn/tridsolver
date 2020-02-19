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

#include "trid_mpi_cpu.hpp"
#include "trid_mpi_cuda.hpp"

#include "trid_linear_mpi_pcr.hpp"
#include "trid_strided_multidim_mpi.hpp"
#include "trid_cuda_mpi_pcr.hpp"

#include "cutil_inline.h"

#include <cassert>
#include <functional>
#include <numeric>
#include <type_traits>
#include <cmath>

#define MAX_REDUCED_LEN 128
#define MIN_TRID_LEN 8

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
                               int sys_n, int num_proc, int mpi_coord, int reducedSysLen) {
  const int reducedSize = reducedSysLen * sys_n;
  
  std::vector<REAL> h_aa_r(reducedSize), h_cc_r(reducedSize), h_dd_r(reducedSize);
  
  REAL *aa_r, *cc_r, *dd_r;
  cudaSafeCall( cudaMalloc(&aa_r, reducedSize * sizeof(REAL)) );
  cudaSafeCall( cudaMalloc(&cc_r, reducedSize * sizeof(REAL)) );
  cudaSafeCall( cudaMalloc(&dd_r, reducedSize * sizeof(REAL)) );
                    
  #pragma omp parallel for
  for (size_t eq_idx = 0; eq_idx < sys_n; ++eq_idx) {
    // The offset in the send and receive buffers
    const size_t buf_offset = eq_idx * 6;
    const size_t buf_size = 6 * sys_n;
    for (int i = 0; i < num_proc; ++i) {
      h_aa_r[eq_idx * num_proc * 2 + (2 * i + 0)] = receive_buf[buf_size * i + buf_offset + 0];
      h_aa_r[eq_idx * num_proc * 2 + (2 * i + 1)] = receive_buf[buf_size * i + buf_offset + 1];
      h_cc_r[eq_idx * num_proc * 2 + (2 * i + 0)] = receive_buf[buf_size * i + buf_offset + 2];
      h_cc_r[eq_idx * num_proc * 2 + (2 * i + 1)] = receive_buf[buf_size * i + buf_offset + 3];
      h_dd_r[eq_idx * num_proc * 2 + (2 * i + 0)] = receive_buf[buf_size * i + buf_offset + 4];
      h_dd_r[eq_idx * num_proc * 2 + (2 * i + 1)] = receive_buf[buf_size * i + buf_offset + 5];
    }
  }
  
  /*
  * Included for debugging
  for(size_t eq_idx = 0; eq_idx < sys_n; ++eq_idx) {
    int ind = eq_idx * reducedSysLen;
    thomas_on_reduced<REAL>(h_aa_r.data() + ind, h_cc_r.data() + ind, h_dd_r.data() + ind, reducedSysLen, 1);
  }
  */
  
  cudaSafeCall( cudaMemcpy(aa_r, h_aa_r.data(), sizeof(REAL) * reducedSize, cudaMemcpyHostToDevice) );
  cudaSafeCall( cudaMemcpy(cc_r, h_cc_r.data(), sizeof(REAL) * reducedSize, cudaMemcpyHostToDevice) );
  cudaSafeCall( cudaMemcpy(dd_r, h_dd_r.data(), sizeof(REAL) * reducedSize, cudaMemcpyHostToDevice) );
  
  // Call PCR
  /*int P = (int) ceil(log2((REAL)reducedSysLen));
  int numBlocks = sys_n;
  int numThreads =  reducedSysLen / 2;
  printf("Blocks %d, Threads %d, P %d, 2^P %d, N %d\n", numBlocks, numThreads, P, 1 << P, reducedSysLen);
  pcr_on_reduced_kernel<REAL><<<numBlocks, numThreads>>>(aa_r, cc_r, dd_r, reducedSysLen, P);*/
  
  
  int P = (int) ceil(log2((REAL)reducedSysLen));
  int numBlocks = sys_n;
  int numThreads =  reducedSysLen;
  //printf("Blocks %d, Threads %d, P %d, 2^P %d, N %d\n", numBlocks, numThreads, P, 1 << P, reducedSysLen);
  pure_pcr_on_reduced_kernel<REAL><<<numBlocks, numThreads>>>(aa_r, cc_r, dd_r, reducedSysLen, P);
  
  cudaSafeCall( cudaPeekAtLastError() );
  cudaSafeCall( cudaDeviceSynchronize() );
  
  // TODO change to kernel that places straight in boundaries array
  cudaSafeCall( cudaMemcpy(h_dd_r.data(), dd_r, sizeof(REAL) * reducedSize, cudaMemcpyDeviceToHost) );
  
  #pragma omp parallel for
  for (size_t eq_idx = 0; eq_idx < sys_n; ++eq_idx) {
    results[2 * eq_idx + 0] = h_dd_r[eq_idx * num_proc * 2 + (2 * mpi_coord + 0)];
    results[2 * eq_idx + 1] = h_dd_r[eq_idx * num_proc * 2 + (2 * mpi_coord + 1)];
  }
  
  cudaSafeCall( cudaFree(aa_r) );
  cudaSafeCall( cudaFree(cc_r) );
  cudaSafeCall( cudaFree(dd_r) );
}

template <typename REAL, int INC>
void tridMultiDimBatchSolveMPI(const MpiSolverParams &params, const REAL *a,
                               int *a_pads, const REAL *b, int *b_pads,
                               const REAL *c, int *c_pads, REAL *d, int *d_pads,
                               REAL *u, int *u_pads, int ndim, int solvedim,
                               int *dims) {
  // TODO paddings!!
  assert(solvedim < ndim);
  assert((
      (std::is_same<REAL, float>::value || std::is_same<REAL, double>::value) &&
      "trid_solve_mpi: only double or float values are supported"));

  // The size of the equations / our domain
  const size_t local_eq_size = dims[solvedim];
  assert(local_eq_size > 2 &&
         "One of the processes has fewer than 2 equations, this is not "
         "supported\n");
  const int eq_stride =
      std::accumulate(dims, dims + solvedim, 1, std::multiplies<int>{});

  // The product of the sizes along the dimensions higher than solve_dim; needed
  // for the iteration later
  const int outer_size = std::accumulate(dims + solvedim + 1, dims + ndim, 1,
                                         std::multiplies<int>{});

  // The number of systems to solve
  const int sys_n = eq_stride * outer_size;

  const MPI_Datatype real_datatype =
      std::is_same<REAL, double>::value ? MPI_DOUBLE : MPI_FLOAT;
  
  // Calculate the minimum length of reduced systems possible
  const int min_reduced_len_g = 2 * params.num_mpi_procs[solvedim];
  // Set the reduced system length to this minimum
  int reduced_len_g = min_reduced_len_g;
  
  // If replacing MAX_REDUCED_LEN with number based on memory size in future, then 
  // should throw error if min_reduced_len > MAX_REDUCED_LEN
  // See if length of reduced system can be increased.
  if(reduced_len_g < MAX_REDUCED_LEN) {
    // TODO
  }
  
  // Calculate how many local elements are part of the global reduced system
  int reduced_len_l = reduced_len_g / params.num_mpi_procs[solvedim];

  const int local_helper_size = outer_size * eq_stride * local_eq_size;
  REAL *aa, *cc, *dd, *boundaries;
  cudaSafeCall( cudaMalloc(&aa, local_helper_size * sizeof(REAL)) );
  cudaSafeCall( cudaMalloc(&cc, local_helper_size * sizeof(REAL)) );
  cudaSafeCall( cudaMalloc(&dd, local_helper_size * sizeof(REAL)) );
  cudaSafeCall( cudaMalloc(&boundaries, sys_n * 3 * reduced_len_l * sizeof(REAL)) );

  int trid_split_factor = reduced_len_g / min_reduced_len_g;
  int total_trids = sys_n * trid_split_factor;
  
  int blockdimx = 128; // Has to be the multiple of 4(or maybe 32??)
  int blockdimy = 1;
  int dimgrid = 1 + (total_trids - 1) / blockdimx; // can go up to 65535
  int dimgridx = dimgrid % 65536;            // can go up to max 65535 on Fermi
  int dimgridy = 1 + dimgrid / 65536;

  dim3 dimGrid_x(dimgridx, dimgridy);
  dim3 dimBlock_x(blockdimx, blockdimy);
  if (solvedim == 0) {
    trid_linear_forward<REAL>
        <<<dimGrid_x, dimBlock_x>>>(a, b, c, d, aa, cc, dd, boundaries,
                                    local_eq_size, local_eq_size, sys_n, trid_split_factor);
    cudaSafeCall( cudaPeekAtLastError() );
    cudaSafeCall( cudaDeviceSynchronize() );
  } else {
    DIM_V pads, dims; // TODO
    for (int i = 0; i < ndim; ++i) {
      pads.v[i] = a_pads[i];
      dims.v[i] = a_pads[i];
    }
    trid_strided_multidim_forward<REAL><<<dimGrid_x, dimBlock_x>>>(
        a, pads, b, pads, c, pads, d, pads, aa, cc, dd, boundaries,
        ndim, solvedim, sys_n, dims);
    cudaSafeCall( cudaPeekAtLastError() );
    cudaSafeCall( cudaDeviceSynchronize() );
  }
  // MPI buffers (6 because 2 from each of the a, c and d coefficient arrays)
  const size_t comm_buf_size = 6 * sys_n;
  std::vector<REAL> send_buf(comm_buf_size),
      receive_buf(comm_buf_size * params.num_mpi_procs[solvedim]);
  cudaSafeCall( cudaMemcpy(send_buf.data(), boundaries, sizeof(REAL) * comm_buf_size,
             cudaMemcpyDeviceToHost) );
  // Communicate boundary results
  MPI_Allgather(send_buf.data(), comm_buf_size, real_datatype,
                receive_buf.data(), comm_buf_size, real_datatype,
                params.communicators[solvedim]);
  // solve reduced systems, and store results in send_buf
  thomas_on_reduced_batched<REAL>(receive_buf.data(), send_buf.data(), sys_n,
                                  params.num_mpi_procs[solvedim],
                                  params.mpi_coords[solvedim], reduced_len_g);

  // copy the results of the reduced systems to the beginning of the boundaries
  // array
  cudaSafeCall( cudaMemcpy(boundaries, send_buf.data(), sizeof(REAL) * 2 * sys_n,
             cudaMemcpyHostToDevice) );

  if (solvedim == 0) {
    trid_linear_backward<REAL, INC><<<dimGrid_x, dimBlock_x>>>(
        aa, cc, dd, d, u, boundaries, local_eq_size, local_eq_size, sys_n);
    cudaSafeCall( cudaPeekAtLastError() );
    cudaSafeCall( cudaDeviceSynchronize() );
  } else {
    DIM_V pads, dims; // TODO
    for (int i = 0; i < ndim; ++i) {
      pads.v[i] = a_pads[i];
      dims.v[i] = a_pads[i];
    }
    trid_strided_multidim_backward<REAL, INC>
        <<<dimGrid_x, dimBlock_x>>>(aa, pads, cc, pads, dd, d, pads, u, pads,
                                    boundaries, ndim, solvedim, sys_n, dims);
    cudaSafeCall( cudaPeekAtLastError() );
    cudaSafeCall( cudaDeviceSynchronize() );
  }

  cudaSafeCall( cudaFree(aa) );
  cudaSafeCall( cudaFree(cc) );
  cudaSafeCall( cudaFree(dd) );
  cudaSafeCall( cudaFree(boundaries) );
}

template <typename REAL, int INC>
void tridMultiDimBatchSolveMPI(const MpiSolverParams &params, const REAL *a,
                               const REAL *b, const REAL *c, REAL *d, REAL *u,
                               int ndim, int solvedim, int *dims, int *pads) {
  tridMultiDimBatchSolveMPI<REAL, INC>(params, a, pads, b, pads, c, pads, d,
                                       pads, u, pads, ndim, solvedim, dims);
}

EXTERN_C
tridStatus_t tridDmtsvStridedBatchMPI(const MpiSolverParams &params,
                                      const double *a, const double *b,
                                      const double *c, double *d, double *u,
                                      int ndim, int solvedim, int *dims,
                                      int *pads) {
  tridMultiDimBatchSolveMPI<double, 0>(params, a, b, c, d, u, ndim, solvedim,
                                       dims, pads);
  return TRID_STATUS_SUCCESS;
}

EXTERN_C
tridStatus_t tridSmtsvStridedBatchMPI(const MpiSolverParams &params,
                                      const float *a, const float *b,
                                      const float *c, float *d, float *u,
                                      int ndim, int solvedim, int *dims,
                                      int *pads) {
  tridMultiDimBatchSolveMPI<float, 0>(params, a, b, c, d, u, ndim, solvedim,
                                      dims, pads);
  return TRID_STATUS_SUCCESS;
}

EXTERN_C
tridStatus_t tridDmtsvStridedBatchIncMPI(const MpiSolverParams &params,
                                      const double *a, const double *b,
                                      const double *c, double *d, double *u,
                                      int ndim, int solvedim, int *dims,
                                      int *pads) {
  tridMultiDimBatchSolveMPI<double, 1>(params, a, b, c, d, u, ndim, solvedim,
                                       dims, pads);
  return TRID_STATUS_SUCCESS;
}

EXTERN_C
tridStatus_t tridSmtsvStridedBatchIncMPI(const MpiSolverParams &params,
                                      const float *a, const float *b,
                                      const float *c, float *d, float *u,
                                      int ndim, int solvedim, int *dims,
                                      int *pads) {
  tridMultiDimBatchSolveMPI<float, 1>(params, a, b, c, d, u, ndim, solvedim,
                                      dims, pads);
  return TRID_STATUS_SUCCESS;
}

EXTERN_C
tridStatus_t tridDmtsvStridedBatchPaddedMPI(
    const MpiSolverParams &params, const double *a, int *a_pads,
    const double *b, int *b_pads, const double *c, int *c_pads, double *d,
    int *d_pads, double *u, int *u_pads, int ndim, int solvedim, int *dims) {
  tridMultiDimBatchSolveMPI<double, 0>(params, a, a_pads, b, b_pads, c, c_pads,
                                       d, d_pads, u, u_pads, ndim, solvedim,
                                       dims);
  return TRID_STATUS_SUCCESS;
}

EXTERN_C
tridStatus_t tridSmtsvStridedBatchPaddedMPI(
    const MpiSolverParams &params, const float *a, int *a_pads, const float *b,
    int *b_pads, const float *c, int *c_pads, float *d, int *d_pads, float *u,
    int *u_pads, int ndim, int solvedim, int *dims) {
  tridMultiDimBatchSolveMPI<float, 0>(params, a, a_pads, b, b_pads, c, c_pads,
                                      d, d_pads, u, u_pads, ndim, solvedim,
                                      dims);
  return TRID_STATUS_SUCCESS;
}

EXTERN_C
tridStatus_t tridDmtsvStridedBatchPaddedIncMPI(
    const MpiSolverParams &params, const double *a, int *a_pads,
    const double *b, int *b_pads, const double *c, int *c_pads, double *d,
    int *d_pads, double *u, int *u_pads, int ndim, int solvedim, int *dims) {
  tridMultiDimBatchSolveMPI<double, 1>(params, a, a_pads, b, b_pads, c, c_pads,
                                       d, d_pads, u, u_pads, ndim, solvedim,
                                       dims);
  return TRID_STATUS_SUCCESS;
}

EXTERN_C
tridStatus_t tridSmtsvStridedBatchPaddedIncMPI(
    const MpiSolverParams &params, const float *a, int *a_pads, const float *b,
    int *b_pads, const float *c, int *c_pads, float *d, int *d_pads, float *u,
    int *u_pads, int ndim, int solvedim, int *dims) {
  tridMultiDimBatchSolveMPI<float, 1>(params, a, a_pads, b, b_pads, c, c_pads,
                                      d, d_pads, u, u_pads, ndim, solvedim,
                                      dims);
  return TRID_STATUS_SUCCESS;
}

