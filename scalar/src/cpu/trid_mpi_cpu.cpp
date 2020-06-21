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

// Written by Toby Flynn, University of Warwick, T.Flynn@warwick.ac.uk, 2020

#include "trid_mpi_cpu.h"

#include "trid_mpi_cpu.hpp"
#include "trid_cpu.h"
#include "trid_mpi_decomposition.hpp"
#include "trid_simd.h"
#include "math.h"
#include "omp.h"

#include <type_traits>
#include <sys/time.h>

#define ROUND_DOWN(N,step) (((N)/(step))*step)
#define MIN(X,Y) ((X) < (Y) ? (X) : (Y))
#define MAX(X,Y) ((X) > (Y) ? (X) : (Y))

#define Z_BATCH 56

template<typename REAL>
void swapDecomposition(MpiSolverParams &params, REAL *ax, REAL *bx, REAL *cx,
                       REAL *dx, REAL *ux, REAL *ay, REAL *by, REAL *cy, REAL *dy,
                       REAL *uy, REAL *az, REAL *bz, REAL *cz, REAL *dz, REAL *uz,
                       int ndim, int solvedim, int *dims, int *pads) {
  int sizesZ[3] = {lsz(dims[0], params.communicators[0]), lsz(dims[1], params.communicators[1]), dims[2]};
  int sizesY[3] = {lsz(dims[0], params.communicators[0]), dims[1], lsz(dims[2], params.communicators[1])};
  int sizesX[3] = {dims[0], lsz(dims[1], params.communicators[0]), lsz(dims[2], params.communicators[1])};

  const MPI_Datatype real_datatype = std::is_same<REAL, double>::value ? MPI_DOUBLE : MPI_FLOAT;

  if(params.currentDim == 0 && solvedim == 1) {
    exchange(params.communicators[0], real_datatype, 3, sizesX, ax, 0, sizesY, ay, 1);
    exchange(params.communicators[0], real_datatype, 3, sizesX, bx, 0, sizesY, by, 1);
    exchange(params.communicators[0], real_datatype, 3, sizesX, cx, 0, sizesY, cy, 1);
    exchange(params.communicators[0], real_datatype, 3, sizesX, dx, 0, sizesY, dy, 1);
    exchange(params.communicators[0], real_datatype, 3, sizesX, ux, 0, sizesY, uy, 1);
    params.currentDim = 1;
    return;
  }

  if(params.currentDim == 1 && solvedim == 2) {
    exchange(params.communicators[1], real_datatype, 3, sizesY, ay, 0, sizesZ, az, 1);
    exchange(params.communicators[1], real_datatype, 3, sizesY, by, 0, sizesZ, bz, 1);
    exchange(params.communicators[1], real_datatype, 3, sizesY, cy, 0, sizesZ, cz, 1);
    exchange(params.communicators[1], real_datatype, 3, sizesY, dy, 0, sizesZ, dz, 1);
    exchange(params.communicators[1], real_datatype, 3, sizesY, uy, 0, sizesZ, uz, 1);
    params.currentDim = 2;
    return;
  }

  if(params.currentDim == 2 && solvedim == 0) {
    exchange(params.communicators[0], real_datatype, 3, sizesZ, az, 0, sizesX, ax, 1);
    exchange(params.communicators[0], real_datatype, 3, sizesZ, bz, 0, sizesX, bx, 1);
    exchange(params.communicators[0], real_datatype, 3, sizesZ, cz, 0, sizesX, cx, 1);
    exchange(params.communicators[0], real_datatype, 3, sizesZ, dz, 0, sizesX, dx, 1);
    exchange(params.communicators[0], real_datatype, 3, sizesZ, uz, 0, sizesX, ux, 1);
    params.currentDim = 0;
    return;
  }
}

template<typename REAL, int INC>
void tridMultiDimBatchSolve(MpiSolverParams &params, REAL *ax, REAL *bx, REAL *cx,
                            REAL *dx, REAL *ux, REAL *ay, REAL *by, REAL *cy, REAL *dy,
                            REAL *uy, REAL *az, REAL *bz, REAL *cz, REAL *dz, REAL *uz,
                            int ndim, int solvedim, int *dims, int *pads) {
  // Swap pencil demcomposition orientation if needed
  if(params.currentDim != solvedim) {
    swapDecomposition<REAL>(params, ax, bx, cx, dx, ux, ay, by, cy, dy, uy, az,
                            bz, cz, dz, uz, ndim, solvedim, dims, pads);
  }

  int sizesZ[3] = {lsz(dims[0], params.communicators[0]), lsz(dims[1], params.communicators[1]), dims[2]};
  int sizesY[3] = {lsz(dims[0], params.communicators[0]), dims[1], lsz(dims[2], params.communicators[1])};
  int sizesX[3] = {dims[0], lsz(dims[1], params.communicators[0]), lsz(dims[2], params.communicators[1])};

  // Solve tridiagonal systems in solvedim
  // Padding not implmented yet
  if(solvedim == 0) {
    if(INC) {
      if(std::is_same<REAL, float>::value) {
        tridSmtsvStridedBatchInc((float *)ax, (float *)bx, (float *)cx, (float *)dx, (float *)ux, ndim, 0, sizesX, sizesX);
      } else {
        tridDmtsvStridedBatchInc((double *)ax, (double *)bx, (double *)cx, (double *)dx, (double *)ux, ndim, 0, sizesX, sizesX);
      }
    } else {
      if(std::is_same<REAL, float>::value) {
        tridSmtsvStridedBatch((float *)ax, (float *)bx, (float *)cx, (float *)dx, (float *)ux, ndim, 0, sizesX, sizesX);
      } else {
        tridDmtsvStridedBatch((double *)ax, (double *)bx, (double *)cx, (double *)dx, (double *)ux, ndim, 0, sizesX, sizesX);
      }
    }
  } else if(solvedim == 1) {
    if(INC) {
      if(std::is_same<REAL, float>::value) {
        tridSmtsvStridedBatchInc((float *)ay, (float *)by, (float *)cy, (float *)dy, (float *)uy, ndim, 1, sizesY, sizesY);
      } else {
        tridDmtsvStridedBatchInc((double *)ay, (double *)by, (double *)cy, (double *)dy, (double *)uy, ndim, 1, sizesY, sizesY);
      }
    } else {
      if(std::is_same<REAL, float>::value) {
        tridSmtsvStridedBatch((float *)ay, (float *)by, (float *)cy, (float *)dy, (float *)uy, ndim, 1, sizesY, sizesY);
      } else {
        tridDmtsvStridedBatch((double *)ay, (double *)by, (double *)cy, (double *)dy, (double *)uy, ndim, 1, sizesY, sizesY);
      }
    }
  } else {
    if(INC) {
      if(std::is_same<REAL, float>::value) {
        tridSmtsvStridedBatchInc((float *)az, (float *)bz, (float *)cz, (float *)dz, (float *)uz, ndim, 2, sizesZ, sizesZ);
      } else {
        tridDmtsvStridedBatchInc((double *)az, (double *)bz, (double *)cz, (double *)dz, (double *)uz, ndim, 2, sizesZ, sizesZ);
      }
    } else {
      if(std::is_same<REAL, float>::value) {
        tridSmtsvStridedBatch((float *)az, (float *)bz, (float *)cz, (float *)dz, (float *)uz, ndim, 2, sizesZ, sizesZ);
      } else {
        tridDmtsvStridedBatch((double *)az, (double *)bz, (double *)cz, (double *)dz, (double *)uz, ndim, 2, sizesZ, sizesZ);
      }
    }
  }
}

// Solve a batch of tridiagonal systems along a specified axis ('solvedim').
// 'a', 'b', 'c', 'd' are the parameters of the tridiagonal systems which must be stored in
// arrays of size 'dims' with 'ndim' dimensions. The 'pads' array specifies any padding used in
// the arrays (the total length of each dimension including padding).
//
// The result is written to 'd'. 'u' is unused.
#if FPPREC == 1
tridStatus_t tridDmtsvStridedBatchMPI(MpiSolverParams &params,
                                      double *ax, double *bx, double *cx, double *dx,
                                      double *ux, double *ay, double *by, double *cy,
                                      double *dy, double *uy, double *az, double *bz,
                                      double *cz, double *dz, double *uz, int ndim,
                                      int solvedim, int *dims, int *pads, int *dims_g) {
  tridMultiDimBatchSolve<double, 0>(params, ax, bx, cx, dx, ux, ay, by, cy, dy, uy,
                                    az, bz, cz, dz, uz, ndim, solvedim, dims, pads);
  return TRID_STATUS_SUCCESS;
}
#else
tridStatus_t tridSmtsvStridedBatchMPI(MpiSolverParams &params,
                                      float *ax, float *bx, float *cx, float *dx,
                                      float *ux, float *ay, float *by, float *cy,
                                      float *dy, float *uy, float *az, float *bz,
                                      float *cz, float *dz, float *uz, int ndim,
                                      int solvedim, int *dims, int *pads, int *dims_g) {
  tridMultiDimBatchSolve<float, 0>(params, ax, bx, cx, dx, ux, ay, by, cy, dy, uy,
                                    az, bz, cz, dz, uz, ndim, solvedim, dims, pads);
  return TRID_STATUS_SUCCESS;
}
#endif

// Solve a batch of tridiagonal systems along a specified axis ('solvedim').
// 'a', 'b', 'c', 'd' are the parameters of the tridiagonal systems which must be stored in
// arrays of size 'dims' with 'ndim' dimensions. The 'pads' array specifies any padding used in
// the arrays (the total length of each dimension including padding).
//
// 'u' is incremented with the results.

#if FPPREC == 1
tridStatus_t tridDmtsvStridedBatchIncMPI(MpiSolverParams &params,
                                      double *ax, double *bx, double *cx, double *dx,
                                      double *ux, double *ay, double *by, double *cy,
                                      double *dy, double *uy, double *az, double *bz,
                                      double *cz, double *dz, double *uz, int ndim,
                                      int solvedim, int *dims, int *pads, int *dims_g) {
  tridMultiDimBatchSolve<double, 1>(params, ax, bx, cx, dx, ux, ay, by, cy, dy, uy,
                                    az, bz, cz, dz, uz, ndim, solvedim, dims, pads);
  return TRID_STATUS_SUCCESS;
}
#else
tridStatus_t tridSmtsvStridedBatchIncMPI(MpiSolverParams &params,
                                      float *ax, float *bx, float *cx, float *dx,
                                      float *ux, float *ay, float *by, float *cy,
                                      float *dy, float *uy, float *az, float *bz,
                                      float *cz, float *dz, float *uz, int ndim,
                                      int solvedim, int *dims, int *pads, int *dims_g) {
  tridMultiDimBatchSolve<float, 1>(params, ax, bx, cx, dx, ux, ay, by, cy, dy, uy,
                                    az, bz, cz, dz, uz, ndim, solvedim, dims, pads);
  return TRID_STATUS_SUCCESS;
}
#endif
