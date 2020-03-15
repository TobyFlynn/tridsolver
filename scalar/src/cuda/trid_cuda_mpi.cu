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

#include "trid_linear_mpi_reg8_double2.hpp"
#include "trid_linear_mpi.hpp"
#include "trid_strided_multidim_mpi.hpp"
#include "trid_cuda_mpi_pcr.hpp"

#include "cutil_inline.h"

#include <cassert>
#include <functional>
#include <numeric>
#include <type_traits>
#include <cmath>

//#define MAX_REDUCED_LEN 1024
//#define MIN_TRID_LEN 8
//#define BLOCKING_FACTOR 32

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

  REAL *aa_r, *cc_r, *dd_r;
  cudaSafeCall( cudaMalloc(&aa_r, reducedSize * sizeof(REAL)) );
  cudaSafeCall( cudaMalloc(&cc_r, reducedSize * sizeof(REAL)) );
  cudaSafeCall( cudaMalloc(&dd_r, reducedSize * sizeof(REAL)) );
  
  int blockdimx = 128; // Has to be the multiple of 4(or maybe 32??)
  int blockdimy = 1;
  int dimgrid = 1 + (sys_n - 1) / blockdimx; // can go up to 65535
  int dimgridx = dimgrid % 65536;            // can go up to max 65535 on Fermi
  int dimgridy = 1 + dimgrid / 65536;

  dim3 dimGrid_x(dimgridx, dimgridy);
  dim3 dimBlock_x(blockdimx, blockdimy);

  pcr_on_reduced_kernel_preproc<<<dimGrid_x, dimBlock_x>>>(receive_buf, aa_r, cc_r, dd_r, 
                                                           sys_n, num_proc, reducedSysLen);
  cudaSafeCall( cudaPeekAtLastError() );
  cudaSafeCall( cudaDeviceSynchronize() );

  // Call PCR
  int P = (int) ceil(log2((REAL)reducedSysLen));
  int numBlocks = sys_n;
  int numThreads =  reducedSysLen;
  pcr_on_reduced_kernel<REAL><<<numBlocks, numThreads>>>(aa_r, cc_r, dd_r, results, 
                                                         mpi_coord, reducedSysLen, P);
  
  cudaSafeCall( cudaPeekAtLastError() );
  cudaSafeCall( cudaDeviceSynchronize() );

  cudaSafeCall( cudaFree(aa_r) );
  cudaSafeCall( cudaFree(cc_r) );
  cudaSafeCall( cudaFree(dd_r) );
}

template <typename REAL, int INC>
void tridMultiDimBatchSolveMPI(const MpiSolverParams &params, const REAL *a,
                               int *a_pads, const REAL *b, int *b_pads,
                               const REAL *c, int *c_pads, REAL *d, int *d_pads,
                               REAL *u, int *u_pads, int ndim, int solvedim,
                               int *dims, int *dims_g) {
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

  int *d_dims = NULL;
  cudaSafeCall( cudaMalloc(&d_dims, 3 * sizeof(int)) );
  cudaSafeCall( cudaMemcpy(d_dims, dims, sizeof(int) * 3, cudaMemcpyHostToDevice) );
  
  // The number of systems to solve
  const int sys_n = eq_stride * outer_size;

  const MPI_Datatype real_datatype =
      std::is_same<REAL, double>::value ? MPI_DOUBLE : MPI_FLOAT;
  
  int reduced_len_g;
  int reduced_len_l;
  
  reduced_len_g = 2 * params.num_mpi_procs[solvedim];
  reduced_len_l = 2;

  const int local_helper_size = outer_size * eq_stride * local_eq_size;
  REAL *aa, *cc, *dd, *boundaries;
  cudaSafeCall( cudaMalloc(&aa, local_helper_size * sizeof(REAL)) );
  cudaSafeCall( cudaMalloc(&cc, local_helper_size * sizeof(REAL)) );
  cudaSafeCall( cudaMalloc(&dd, local_helper_size * sizeof(REAL)) );
  cudaSafeCall( cudaMalloc(&boundaries, sys_n * 3 * reduced_len_l * sizeof(REAL)) );
  
  int blockdimx = 128; // Has to be the multiple of 4(or maybe 32??)
  int blockdimy = 1;
  int dimgrid = 1 + (sys_n - 1) / blockdimx; // can go up to 65535
  int dimgridx = dimgrid % 65536;            // can go up to max 65535 on Fermi
  int dimgridy = 1 + dimgrid / 65536;

  dim3 dimGrid_x(dimgridx, dimgridy);
  dim3 dimBlock_x(blockdimx, blockdimy);
  if (solvedim == 0) {
    if(std::is_same<REAL, double>::value) {
      trid_linear_forward
          <<<dimGrid_x, dimBlock_x>>>((double *)a, (double *)b, (double *)c, (double *)d, (double *)aa, (double *)cc, (double *)dd, (double *)boundaries,
                                      local_eq_size, local_eq_size, sys_n);
      cudaSafeCall( cudaPeekAtLastError() );
      cudaSafeCall( cudaDeviceSynchronize() );
    } else {
      // TODO
    }
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
  const size_t comm_buf_size = reduced_len_l * 3 * sys_n;
  /*std::vector<REAL> send_buf(comm_buf_size),
      receive_buf(reduced_len_g * 3 * sys_n);
  cudaSafeCall( cudaMemcpy(send_buf.data(), boundaries, sizeof(REAL) * comm_buf_size,
             cudaMemcpyDeviceToHost) );*/
  REAL *recv_buf;
  cudaSafeCall( cudaMalloc(&recv_buf, reduced_len_g * 3 * sys_n * sizeof(REAL)) );
  
  // Communicate boundary results
  MPI_Allgather(boundaries, comm_buf_size, real_datatype,
                recv_buf, comm_buf_size, real_datatype,
                params.communicators[solvedim]);
  
  thomas_on_reduced_batched<REAL>(receive_buf, boundaries, sys_n, 
                                    params.num_mpi_procs[solvedim],
                                    params.mpi_coords[solvedim], reduced_len_g);

  if (solvedim == 0) {
    if(std::is_same<REAL, double>::value) {
      trid_linear_backward<INC><<<dimGrid_x, dimBlock_x>>>(
          (double *)aa, (double *)cc, (double *)dd, (double *)d, (double *)u, (double *)boundaries, local_eq_size, local_eq_size, sys_n);
      cudaSafeCall( cudaPeekAtLastError() );
      cudaSafeCall( cudaDeviceSynchronize() );
    } else {
      // TODO
    }
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
  cudaSafeCall( cudaFree(d_dims) );
}

template <typename REAL, int INC>
void tridMultiDimBatchSolveMPI(const MpiSolverParams &params, const REAL *a,
                               const REAL *b, const REAL *c, REAL *d, REAL *u,
                               int ndim, int solvedim, int *dims, int *dims_g, int *pads) {
  tridMultiDimBatchSolveMPI<REAL, INC>(params, a, pads, b, pads, c, pads, d,
                                       pads, u, pads, ndim, solvedim, dims, dims_g);
}

EXTERN_C
tridStatus_t tridDmtsvStridedBatchMPI(const MpiSolverParams &params,
                                      const double *a, const double *b,
                                      const double *c, double *d, double *u,
                                      int ndim, int solvedim, int *dims, int *dims_g,
                                      int *pads) {
  tridMultiDimBatchSolveMPI<double, 0>(params, a, b, c, d, u, ndim, solvedim,
                                       dims, dims_g, pads);
  return TRID_STATUS_SUCCESS;
}

EXTERN_C
tridStatus_t tridSmtsvStridedBatchMPI(const MpiSolverParams &params,
                                      const float *a, const float *b,
                                      const float *c, float *d, float *u,
                                      int ndim, int solvedim, int *dims, int *dims_g,
                                      int *pads) {
  tridMultiDimBatchSolveMPI<float, 0>(params, a, b, c, d, u, ndim, solvedim,
                                      dims, dims_g, pads);
  return TRID_STATUS_SUCCESS;
}

EXTERN_C
tridStatus_t tridDmtsvStridedBatchIncMPI(const MpiSolverParams &params,
                                      const double *a, const double *b,
                                      const double *c, double *d, double *u,
                                      int ndim, int solvedim, int *dims, int *dims_g,
                                      int *pads) {
  tridMultiDimBatchSolveMPI<double, 1>(params, a, b, c, d, u, ndim, solvedim,
                                       dims, dims_g, pads);
  return TRID_STATUS_SUCCESS;
}

EXTERN_C
tridStatus_t tridSmtsvStridedBatchIncMPI(const MpiSolverParams &params,
                                      const float *a, const float *b,
                                      const float *c, float *d, float *u,
                                      int ndim, int solvedim, int *dims, int *dims_g,
                                      int *pads) {
  tridMultiDimBatchSolveMPI<float, 1>(params, a, b, c, d, u, ndim, solvedim,
                                      dims, dims_g, pads);
  return TRID_STATUS_SUCCESS;
}

EXTERN_C
tridStatus_t tridDmtsvStridedBatchPaddedMPI(
    const MpiSolverParams &params, const double *a, int *a_pads,
    const double *b, int *b_pads, const double *c, int *c_pads, double *d,
    int *d_pads, double *u, int *u_pads, int ndim, int solvedim, int *dims, int *dims_g) {
  tridMultiDimBatchSolveMPI<double, 0>(params, a, a_pads, b, b_pads, c, c_pads,
                                       d, d_pads, u, u_pads, ndim, solvedim,
                                       dims, dims_g);
  return TRID_STATUS_SUCCESS;
}

EXTERN_C
tridStatus_t tridSmtsvStridedBatchPaddedMPI(
    const MpiSolverParams &params, const float *a, int *a_pads, const float *b,
    int *b_pads, const float *c, int *c_pads, float *d, int *d_pads, float *u,
    int *u_pads, int ndim, int solvedim, int *dims, int *dims_g) {
  tridMultiDimBatchSolveMPI<float, 0>(params, a, a_pads, b, b_pads, c, c_pads,
                                      d, d_pads, u, u_pads, ndim, solvedim,
                                      dims, dims_g);
  return TRID_STATUS_SUCCESS;
}

EXTERN_C
tridStatus_t tridDmtsvStridedBatchPaddedIncMPI(
    const MpiSolverParams &params, const double *a, int *a_pads,
    const double *b, int *b_pads, const double *c, int *c_pads, double *d,
    int *d_pads, double *u, int *u_pads, int ndim, int solvedim, int *dims, int *dims_g) {
  tridMultiDimBatchSolveMPI<double, 1>(params, a, a_pads, b, b_pads, c, c_pads,
                                       d, d_pads, u, u_pads, ndim, solvedim,
                                       dims, dims_g);
  return TRID_STATUS_SUCCESS;
}

EXTERN_C
tridStatus_t tridSmtsvStridedBatchPaddedIncMPI(
    const MpiSolverParams &params, const float *a, int *a_pads, const float *b,
    int *b_pads, const float *c, int *c_pads, float *d, int *d_pads, float *u,
    int *u_pads, int ndim, int solvedim, int *dims, int *dims_g) {
  tridMultiDimBatchSolveMPI<float, 1>(params, a, a_pads, b, b_pads, c, c_pads,
                                      d, d_pads, u, u_pads, ndim, solvedim,
                                      dims, dims_g);
  return TRID_STATUS_SUCCESS;
}

