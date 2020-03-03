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

#ifndef TRID_STRIDED_MULTIDIM_PCR_GPU_MPI__
#define TRID_STRIDED_MULTIDIM_PCR_GPU_MPI__

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
    const REAL *__restrict__ a, const REAL *__restrict__ b,
    const REAL *__restrict__ c, const REAL *__restrict__ d, REAL *__restrict__ aa,
    REAL *__restrict__ cc, REAL *__restrict__ dd, REAL *__restrict__ boundaries,
    int ind, int stride, int ind_bound, int sys_size) {
  //
  // forward pass
  //
  REAL bb;
  for (int i = 0; i < 2; ++i) {
    bb = static_cast<REAL>(1.0) / b[ind + i * stride];
    cc[ind + i * stride] = bb * c[ind + i * stride];
    aa[ind + i * stride] = bb * a[ind + i * stride];
    dd[ind + i * stride] = bb * d[ind + i * stride];
  }

  if (sys_size >= 3) {
    // Eliminate lower off-diagonal
    for (int i = 2; i < sys_size; ++i) {
      bb = static_cast<REAL>(1.0) /
           (b[ind + i * stride] -
            a[ind + i * stride] * cc[ind + (i - 1) * stride]);
      dd[ind + i * stride] =
          (d[ind + i * stride] -
           a[ind + i * stride] * dd[ind + (i - 1) * stride]) *
          bb;
      aa[ind + i * stride] =
          (-a[ind + i * stride] * aa[ind + (i - 1) * stride]) * bb;
      cc[ind + i * stride] = c[ind + i * stride] * bb;
    }
    // Eliminate upper off-diagonal
    for (int i = sys_size - 3; i > 0; --i) {
      dd[ind + i * stride] =
          dd[ind + i * stride] -
          cc[ind + i * stride] * dd[ind + (i + 1) * stride];
      aa[ind + i * stride] =
          aa[ind + i * stride] -
          cc[ind + i * stride] * aa[ind + (i + 1) * stride];
      cc[ind + i * stride] =
          -cc[ind + i * stride] * cc[ind + (i + 1) * stride];
    }
    bb = static_cast<REAL>(1.0) /
         (static_cast<REAL>(1.0) - cc[ind] * aa[ind + stride]);
    dd[ind] = bb * (dd[ind] - cc[ind] * dd[ind + stride]);
    aa[ind] = bb * aa[ind];
    cc[ind] = bb * (-cc[ind] * cc[ind + stride]);
  }
  // prepare boundaries for communication
  boundaries[ind_bound + 0] = aa[ind];
  boundaries[ind_bound + 1] = aa[ind + (sys_size - 1) * stride];
  boundaries[ind_bound + 2] = cc[ind];
  boundaries[ind_bound + 3] = cc[ind + (sys_size - 1) * stride];
  boundaries[ind_bound + 4] = dd[ind];
  boundaries[ind_bound + 5] = dd[ind + (sys_size - 1) * stride];
}

template <typename REAL>//, int BLOCKING_FACTOR>
__global__ void trid_strided_multidim_forward(
    const REAL *__restrict__ a, const REAL *__restrict__ b,
    const REAL *__restrict__ c, const REAL *__restrict__ d, REAL *__restrict__ aa,
    REAL *__restrict__ cc, REAL *__restrict__ dd, REAL *__restrict__ boundaries,
    int solvedim, int sys_n, const int *dims, int split_factor) {
  //
  // set up indices for main block
  //
  // Thread ID in global scope - every thread solves one system
  int tid = threadIdx.x + threadIdx.y * blockDim.x +
        blockIdx.x * blockDim.y * blockDim.x +
        blockIdx.y * gridDim.x * blockDim.y * blockDim.x;
  //int tridNum = tid % sys_n;
  //int subTridNum = tid / sys_n;
  int tridNum = (tid % BLOCKING_FACTOR) + (tid / (BLOCKING_FACTOR * split_factor)) * BLOCKING_FACTOR;
  int subTridNum = (tid % (BLOCKING_FACTOR * split_factor)) / BLOCKING_FACTOR;

  int ind;
  int stride;
  
  if(solvedim == 1) {
    ind = (tridNum / dims[0]) * dims[0] * dims[1] + (tridNum % dims[0]);
    stride = dims[0];
  } else {
    ind = tridNum;
    stride = dims[0] * dims[1];
  }
  
  int ind_bound = (tridNum * split_factor + subTridNum) * 6;
  int sys_size = dims[solvedim];
  
  int totalTrids = sys_n * split_factor;
  int offset = subTridNum * (sys_size / split_factor);
  int len;
  
  if(subTridNum == split_factor - 1) {
    len = sys_size - offset;
  } else {
    len = sys_size / split_factor;
  }
  
  ind += offset * stride;

  if (tid < totalTrids) {
    trid_strided_multidim_forward_kernel<REAL>(
        a, b, c, d, aa, cc, dd, boundaries, ind, stride, ind_bound, len);
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
template <typename REAL, int INC>
__device__ void trid_strided_multidim_backward_kernel(
    const REAL *__restrict__ aa, const REAL *__restrict__ cc,
    const REAL *__restrict__ dd, REAL *__restrict__ d, REAL *__restrict__ u,
    const REAL *__restrict__ boundaries, int ind, int stride, int ind_bound, int sys_size) {
  //
  // reverse pass
  //
  REAL dd0 = boundaries[ind_bound], dd_last = boundaries[ind_bound + 1];
  if(INC==0) d[ind]  = dd0;
  else       u[ind] += dd0;
  for (int i = 1; i < sys_size - 1; i++) {
    REAL res = dd[ind + i * stride] - aa[ind + i * stride] * dd0 -
               cc[ind + i * stride] * dd_last;
    if(INC==0)  d[ind + i * stride] = res;
    else        u[ind + i * stride] += res; 
  }
  if(INC==0) d[ind + (sys_size - 1) * stride]  = dd_last;
  else       u[ind + (sys_size - 1) * stride] += dd_last;
}

template <typename REAL/*, int BLOCKING_FACTOR*/, int INC>
__global__ void
trid_strided_multidim_backward(const REAL *__restrict__ aa, const REAL *__restrict__ cc,
                               const REAL *__restrict__ dd, REAL *__restrict__ d,
                               REAL *__restrict__ u, const REAL *__restrict__ boundaries, 
                               int solvedim, int sys_n, const int *dims, int split_factor) {
  //
  // set up indices for main block
  //
  // Thread ID in global scope - every thread solves one system
  int tid = threadIdx.x + threadIdx.y * blockDim.x +
        blockIdx.x * blockDim.y * blockDim.x +
        blockIdx.y * gridDim.x * blockDim.y * blockDim.x;

  //int tridNum = tid % sys_n;
  //int subTridNum = tid / sys_n;
        
  int tridNum = (tid % BLOCKING_FACTOR) + (tid / (BLOCKING_FACTOR * split_factor)) * BLOCKING_FACTOR;
  int subTridNum = (tid % (BLOCKING_FACTOR * split_factor)) / BLOCKING_FACTOR;
  
  int ind;
  int stride;
  
  if(solvedim == 1) {
    ind = (tridNum / dims[0]) * dims[0] * dims[1] + (tridNum % dims[0]);
    stride = dims[0];
  } else {
    ind = tridNum;
    stride = dims[0] * dims[1];
  }
  
  int ind_bound = (tridNum * split_factor + subTridNum) * 2;
  int sys_size = dims[solvedim];
  
  int totalTrids = sys_n * split_factor;
  int offset = subTridNum * (sys_size / split_factor);
  int len;
  
  if(subTridNum == split_factor - 1) {
    len = sys_size - offset;
  } else {
    len = sys_size / split_factor;
  }
  
  ind += offset * stride;

  if (tid < totalTrids) {
    trid_strided_multidim_backward_kernel<REAL, INC>(
        aa, cc, dd, d, u, boundaries, ind, stride, ind_bound, len);
  }
}

#endif /* ifndef TRID_STRIDED_MULTIDIM_GPU_MPI__ */
