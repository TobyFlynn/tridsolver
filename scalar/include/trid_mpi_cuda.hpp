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

#define N_MPI_MAX 128
#include "trid_common.h"
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

typedef struct {
  int v[8];
} DIM_V;

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

