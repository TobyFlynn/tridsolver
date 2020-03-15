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

#ifndef __TRID_CUDA_MPI_PCR_HPP
#define __TRID_CUDA_MPI_PCR_HPP

template<typename REAL>
__global__ void pcr_on_reduced_kernel(REAL *a, REAL *c, REAL *d, REAL *results, 
                                      const int mpi_coord, const int n, const int P) {
  int tridNum = blockIdx.x;
  int i = threadIdx.x;
  int ind = tridNum * n + i;

  if(i >= n) {

    for(int p = 0; p < P; p++) {
      __syncthreads();
      __syncthreads();
    }
    
    return;
  }
  
  REAL a_m, a_p, c_m, c_p, d_m, d_p;

  int s = 1;
  
  for(int p = 0; p < P; p++) {
    if(i - s < 0) {
      a_m = (REAL) 0.0;
      c_m = (REAL) 0.0;
      d_m = (REAL) 0.0;
    } else {
      a_m = a[ind - s];
      c_m = c[ind - s];
      d_m = d[ind - s];
    }
    
    if(i + s >= n) {
      a_p = (REAL) 0.0;
      c_p = (REAL) 0.0;
      d_p = (REAL) 0.0;
    } else {
      a_p = a[ind + s];
      c_p = c[ind + s];
      d_p = d[ind + s];
    }
    
    __syncthreads();
    
    REAL r = 1.0 - a[ind] * c_m - c[ind] * a_p;
    r = 1.0 / r;
    d[ind] = r * (d[ind] - a[ind] * d_m - c[ind] * d_p);
    a[ind] = -r * a[ind] * a_m;
    c[ind] = -r * c[ind] * c_p;
    
    __syncthreads();
    
    s = s << 1;
  }
  
  if(i >= 2 * mpi_coord && i < 2 * (mpi_coord + 1)) {
    int reduced_ind_l = i - (2 * mpi_coord);
    results[2 * tridNum + reduced_ind_l] = d[ind];
  }
}

template<typename REAL>
__global__ void pcr_on_reduced_kernel_preproc(REAL* input, REAL *a, REAL *c, REAL *d, 
                                      const int sys_n, const int procs, int reducedLen) {
  const int tid = threadIdx.x + threadIdx.y * blockDim.x +
                  blockIdx.x * blockDim.y * blockDim.x +
                  blockIdx.y * gridDim.x * blockDim.y * blockDim.x;

  if(tid < sys_n) {
    int buf_size = 6 * sys_n;
    int buf_offset = tid * 6;
    for(int p = 0; p < procs; p++) {
      a[tid * reducedLen + (2 * p)]     = input[buf_size * p + buf_offset];
      a[tid * reducedLen + (2 * p) + 1] = input[buf_size * p + buf_offset + 1];
      c[tid * reducedLen + (2 * p)]     = input[buf_size * p + buf_offset + 2];
      c[tid * reducedLen + (2 * p) + 1] = input[buf_size * p + buf_offset + 3];
      d[tid * reducedLen + (2 * p)]     = input[buf_size * p + buf_offset + 4];
      d[tid * reducedLen + (2 * p) + 1] = input[buf_size * p + buf_offset + 5];
    }
  }
}

#endif
