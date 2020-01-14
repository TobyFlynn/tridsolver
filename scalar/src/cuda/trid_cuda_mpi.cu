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

#include "trid_linear_mpi.hpp"
#include "trid_strided_multidim_mpi.hpp"
#include <cassert>
#include <functional>
#include <numeric>

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
    std::vector<REAL> aa_r(2 * num_proc), cc_r(2 * num_proc),
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
    thomas_on_reduced<REAL>(aa_r.data(), cc_r.data(), dd_r.data(), 2 * num_proc,
                            1);

    results[2 * eq_idx + 0] = dd_r[2 * mpi_coord + 0];
    results[2 * eq_idx + 1] = dd_r[2 * mpi_coord + 1];
  }
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
  if (solvedim == 0) {
    trid_linear_forward<REAL>
        <<<dimGrid_x, dimBlock_x>>>(a, b, c, d, aa, cc, dd, boundaries,
                                    local_eq_size, local_eq_size, sys_n);
  } else {
    DIM_V pads, dims; // TODO
    for (int i = 0; i < ndim; ++i) {
      pads.v[i] = a_pads[i];
      dims.v[i] = a_pads[i];
    }
    trid_strided_multidim_forward<REAL><<<dimGrid_x, dimBlock_x>>>(
        a, pads, b, pads, c, pads, d, pads, aa, cc, dd, boundaries,
        ndim, solvedim, sys_n, dims);
  }
  // MPI buffers (6 because 2 from each of the a, c and d coefficient arrays)
  const size_t comm_buf_size = 6 * sys_n;
  std::vector<REAL> send_buf(comm_buf_size),
      receive_buf(comm_buf_size * params.num_mpi_procs[solvedim]);
  cudaMemcpy(send_buf.data(), boundaries, sizeof(REAL) * comm_buf_size,
             cudaMemcpyDeviceToHost);
  // Communicate boundary results
  MPI_Allgather(send_buf.data(), comm_buf_size, real_datatype,
                receive_buf.data(), comm_buf_size, real_datatype,
                params.communicators[solvedim]);
  // solve reduced systems, and store results in send_buf
  thomas_on_reduced_batched<REAL>(receive_buf.data(), send_buf.data(), sys_n,
                                  params.num_mpi_procs[solvedim],
                                  params.mpi_coords[solvedim]);

  // copy the results of the reduced systems to the beginning of the boundaries
  // array
  cudaMemcpy(boundaries, send_buf.data(), sizeof(REAL) * 2 * sys_n,
             cudaMemcpyHostToDevice);

  if (solvedim == 0) {
    trid_linear_backward<REAL, INC><<<dimGrid_x, dimBlock_x>>>(
        aa, cc, dd, d, u, boundaries, local_eq_size, local_eq_size, sys_n);
  } else {
    DIM_V pads, dims; // TODO
    for (int i = 0; i < ndim; ++i) {
      pads.v[i] = a_pads[i];
      dims.v[i] = a_pads[i];
    }
    trid_strided_multidim_backward<REAL, INC>
        <<<dimGrid_x, dimBlock_x>>>(aa, pads, cc, pads, dd, d, pads, u, pads,
                                    boundaries, ndim, solvedim, sys_n, dims);
  }

  cudaFree(aa);
  cudaFree(cc);
  cudaFree(dd);
  cudaFree(boundaries);
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

