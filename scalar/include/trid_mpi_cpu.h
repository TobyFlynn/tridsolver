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

#ifndef __TRID_MPI_CPU_H
#define __TRID_MPI_CPU_H

#include "mpi.h"

#include "trid_common.h"
#include "trid_mpi_solver_params.hpp"

// Solve a batch of tridiagonal systems along a specified axis ('solvedim').
// 'a', 'b', 'c', 'd' are the parameters of the tridiagonal systems which must be stored in
// arrays of size 'dims' with 'ndim' dimensions. The 'pads' array specifies any padding used in
// the arrays (the total length of each dimension including padding).
//
// The result is written to 'd'. 'u' is unused.
tridStatus_t tridDmtsvStridedBatchMPI(MpiSolverParams &params,
                                      double *ax, double *bx, double *cx, double *dx,
                                      double *ux, double *ay, double *by, double *cy,
                                      double *dy, double *uy, double *az, double *bz,
                                      double *cz, double *dz, double *uz, int ndim,
                                      int solvedim, int *dims, int *pads, int *dims_g);

tridStatus_t tridSmtsvStridedBatchMPI(MpiSolverParams &params,
                                      float *ax, float *bx, float *cx, float *dx,
                                      float *ux, float *ay, float *by, float *cy,
                                      float *dy, float *uy, float *az, float *bz,
                                      float *cz, float *dz, float *uz, int ndim,
                                      int solvedim, int *dims, int *pads, int *dims_g);

// Solve a batch of tridiagonal systems along a specified axis ('solvedim').
// 'a', 'b', 'c', 'd' are the parameters of the tridiagonal systems which must be stored in
// arrays of size 'dims' with 'ndim' dimensions. The 'pads' array specifies any padding used in
// the arrays (the total length of each dimension including padding).
//
// 'u' is incremented with the results.
tridStatus_t tridDmtsvStridedBatchIncMPI(MpiSolverParams &params,
                                      double *ax, double *bx, double *cx, double *dx,
                                      double *ux, double *ay, double *by, double *cy,
                                      double *dy, double *uy, double *az, double *bz,
                                      double *cz, double *dz, double *uz, int ndim,
                                      int solvedim, int *dims, int *pads, int *dims_g);

tridStatus_t tridSmtsvStridedBatchIncMPI(MpiSolverParams &params,
                                      float *ax, float *bx, float *cx, float *dx,
                                      float *ux, float *ay, float *by, float *cy,
                                      float *dy, float *uy, float *az, float *bz,
                                      float *cz, float *dz, float *uz, int ndim,
                                      int solvedim, int *dims, int *pads, int *dims_g);

#endif
