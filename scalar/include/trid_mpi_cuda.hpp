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
//
#ifndef TRID_MPI_GPU_HPP__
#define TRID_MPI_GPU_HPP__

#define N_MPI_MAX 128
#include "trid_common.h"
#include "trid_mpi_cpu.hpp"

#include "trid_cuda.h"

template <typename REAL>
__global__ void
trid_linear_forward(const REAL *__restrict__ a, const REAL *__restrict__ b,
                    const REAL *__restrict__ c, const REAL *__restrict__ d,
                    const REAL *__restrict__ u, REAL *__restrict__ aa,
                    REAL *__restrict__ cc, REAL *__restrict__ dd,
                    REAL *__restrict__ boundaries, int sys_size, int sys_pads,
                    int sys_n);

template <typename REAL>
__global__ void
trid_linear_backward(const REAL *__restrict__ aa, const REAL *__restrict__ cc,
                     const REAL *__restrict__ dd, REAL *__restrict__ d,
                     REAL *__restrict__ u, const REAL *__restrict__ boundaries,
                     int sys_size, int sys_pads, int sys_n);

typedef struct {
  int v[8];
} DIM_V;

template <typename REAL>
__global__ void trid_strided_multidim_forward(
    const REAL *__restrict__ a, const DIM_V a_pads, const REAL *__restrict__ b,
    const DIM_V b_pads, const REAL *__restrict__ c, const DIM_V c_pads,
    const REAL *__restrict__ d, const DIM_V d_pads, const REAL *__restrict__ u,
    const DIM_V u_pads, REAL *__restrict__ aa, REAL *__restrict__ cc,
    REAL *__restrict__ dd, REAL *__restrict__ boundaries, int ndim,
    int solvedim, int sys_n, const DIM_V dims);


template <typename REAL>
__device__ void trid_strided_multidim_backward_kernel(
    const REAL *__restrict__ aa, int ind_a, int stride_a,
    const REAL *__restrict__ cc, int ind_c, int stride_c,
    const REAL *__restrict__ dd, REAL *__restrict__ d, int ind_d, int stride_d,
    REAL *__restrict__ u, int ind_u, int stride_u,
    const REAL *__restrict__ boundaries, int ind_bound, int sys_size);

template <typename REAL>
__global__ void
trid_strided_multidim_backward(const REAL *__restrict__ aa, const DIM_V a_pads,
                               const REAL *__restrict__ cc, const DIM_V c_pads,
                               const REAL *__restrict__ dd,
                               REAL *__restrict__ d, const DIM_V d_pads,
                               REAL *__restrict__ u, const DIM_V u_pads,
                               const REAL *__restrict__ boundaries, int ndim,
                               int solvedim, int sys_n, const DIM_V dims);
template <typename REAL>
void thomas_on_reduced_batched(const REAL *receive_buf, REAL *results,
                               int sys_n, int num_proc, int mpi_coord);

template <typename REAL>
void tridMultiDimBatchSolveMPI(const MpiSolverParams &params, const REAL *a,
                               int *a_pads, const REAL *b, int *b_pads,
                               const REAL *c, int *c_pads, REAL *d, int *d_pads,
                               REAL *u, int *u_pads);

template <typename REAL>
void tridMultiDimBatchSolveMPI(const MpiSolverParams &params, const REAL *a,
                               const REAL *b, const REAL *c, REAL *d, REAL *u,
                               int *pads);


EXTERN_C
tridStatus_t tridDmtsvStridedBatchMPI(const MpiSolverParams &params,
                                      const double *a, const double *b,
                                      const double *c, double *d, double *u,
                                      int *pads);

EXTERN_C
tridStatus_t tridSmtsvStridedBatchMPI(const MpiSolverParams &params,
                                      const float *a, const float *b,
                                      const float *c, float *d, float *u,
                                      int *pads);

EXTERN_C
tridStatus_t tridDmtsvStridedBatchPaddedMPI(const MpiSolverParams &params,
                                            const double *a, int *a_pads,
                                            const double *b, int *b_pads,
                                            const double *c, int *c_pads,
                                            double *d, int *d_pads, double *u,
                                            int *u_pads);

EXTERN_C
tridStatus_t tridSmtsvStridedBatchPaddedMPI(const MpiSolverParams &params,
                                            const float *a, int *a_pads,
                                            const float *b, int *b_pads,
                                            const float *c, int *c_pads,
                                            float *d, int *d_pads, float *u,
                                            int *u_pads);

#endif
