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
__device__ void pcr_step(REAL *a, REAL *c, REAL *d, const int n, const int s) {
  int tridNum = blockIdx.x;
  int i = threadIdx.x * 2;
  int ind = tridNum * n + i;
  
  REAL a_m, a_i, a_p;
  REAL c_m, c_i, c_p;
  REAL d_m, d_i, d_p;
  
  a_i = a[ind];
  c_i = c[ind];
  d_i = d[ind];
  
  if(i - s < 0) {
    a_m = (REAL) 0.0;
    c_m = (REAL) 0.0;
    d_m = (REAL) 0.0;
  } else {
    a_m = a[ind - s];
    c_m = a[ind - s];
    d_m = a[ind - s];
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
  
  REAL r = 1.0 - a_i * c_m - c_i * a_p;
  r = 1.0 / r;
  d[ind] = r * (d_i - a_i * d_m - c_i * d_p);
  a[ind] = -r * a_i * a_m;
  c[ind] = -r * c_i * c_p;
}

/*
 * For now assume one trid per block, with (number of elements) / 2 threads per block
 */
template<typename REAL>
__global__ void pcr_on_reduced_kernel(REAL *a, REAL *c, REAL *d, const int n, const int P) {
  int tridNum = blockIdx.x;
  int i = threadIdx.x * 2;
  int ind = tridNum * n + i;
  
  REAL a_m, a_i, a_p;
  REAL c_m, c_i, c_p;
  REAL d_m, d_i, d_p;
  
  a_i = a[ind];
  c_i = c[ind];
  d_i = d[ind];
  
  if(i - 1 < 0) {
    a_m = (REAL) 0.0;
    c_m = (REAL) 0.0;
    d_m = (REAL) 0.0;
    
    a_p = a[ind + 1];
    c_p = c[ind + 1];
    d_p = d[ind + 1];
  } else if(i + 1 >= n) {
    a_m = a[ind - 1];
    c_m = a[ind - 1];
    d_m = a[ind - 1];
    
    a_p = (REAL) 0.0;
    c_p = (REAL) 0.0;
    d_p = (REAL) 0.0;
  } else {
    a_m = a[ind - 1];
    c_m = a[ind - 1];
    d_m = a[ind - 1];
    
    a_p = a[ind + 1];
    c_p = c[ind + 1];
    d_p = d[ind + 1];
  }
  
  // Potentially not needed for first step? 
  // But might improve performance if cache has to be updated?
  __syncthreads();
  
  REAL r = 1.0 - a_i * c_m - c_i * a_p;
  r = 1.0 / r;
  d[ind] = r * (d_i - a_i * d_m - c_i * d_p);
  a[ind] = -r * a_i * a_m;
  c[ind] = -r * c_i * c_p;
  
  __syncthreads();
  
  int s = 1;
  for(int p = 1; p < P; p++) {
    s = s << 1;
    pcr_step<REAL>(a, c, d, n, s);
    
    __syncthreads();
  }
  
  if(i + 1 >= n) {
    d_i = d[ind];
    
    __syncthreads();
    
    d[ind] = d_i - a[ind] * d_i;
  } else {
    REAL d_i1, d_i2;
    d_i = d[ind];
    d_i1 = d[ind + 1];
    if(i + 2 >= n) {
      d_i2 = (REAL) 0.0;
    } else {
      d_i2 = d[ind + 2];
    }
    
    __syncthreads();
    
    d[ind] = d_i - a[ind] * d_i - c[ind] * d_i1;
    d[ind + 1] = d_i1 - a[ind + 1] * d_i1 - c[ind + 1] * d_i2;
  }
}

#endif
