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

// Written by Endre Laszlo, University of Oxford, endre.laszlo@oerc.ox.ac.uk, 2013-2014 
 
#ifndef __TRID_MULTIDIM_H
#define __TRID_MULTIDIM_H

#include "helper_math.h"

//
// Tridiagonal solver for multidimensional batch problems
//
template <typename REAL, typename VECTOR, int INC>
__device__ void trid_strided_multidim_kernel(const VECTOR* __restrict__ a,
                                      const VECTOR* __restrict__ b,
                                      const VECTOR* __restrict__ c,
                                      VECTOR* __restrict__ d,
                                      VECTOR* __restrict__ u,
                                      int ind, int stride, int sys_size) {
   VECTOR aa, bb, cc, dd, c2[N_MAX], d2[N_MAX];
   //
   // forward pass
   //
   bb    = (static_cast<REAL>(1.0))  / b[ind];
   cc    = bb*c[ind];
   dd    = bb*d[ind];
   c2[0] = cc;
   d2[0] = dd;
   for(int j=1; j<sys_size; j++) {
      ind   = ind + stride;
      aa    = a[ind];
      bb    = b[ind] - aa*cc;
      dd    = d[ind] - aa*dd;
      bb    = (static_cast<REAL>(1.0))  / bb;
      cc    = bb*c[ind];
      dd    = bb*dd;
      c2[j] = cc;
      d2[j] = dd;
   }
   //
   // reverse pass
   //
   if(INC==0) d[ind]  = dd;
   else       u[ind] += dd;
   //u[ind] = dd;
   for(int j=sys_size-2; j>=0; j--) {
      ind    = ind - stride;
      dd     = d2[j] - c2[j]*dd;
      if(INC==0) d[ind]  = dd;
      else       u[ind] += dd;
   }
}

struct int8 {
   int v[MAXDIM];
};


template<typename REAL, typename VECTOR, int INC>
__global__ void trid_strided_multidim(const VECTOR* __restrict__ a,
      const VECTOR* __restrict__ b,
      const VECTOR* __restrict__ c,
      VECTOR* __restrict__ d,
      VECTOR* __restrict__ u, int ndim,
      int solvedim, int sys_n,
      const int8 dims, const int8 pads
      ) {

   int tid = threadIdx.x + threadIdx.y*blockDim.x + threadIdx.z*blockDim.x*blockDim.y;
   if ( solvedim < 1 || solvedim > ndim ) return; /* Just hints to the compiler */

   int __shared__ d_cumpads[MAXDIM + 1];
   int __shared__ d_cumdims[MAXDIM + 1];


   if ( tid == 0 ) {

      d_cumdims[0] = d_cumpads[0] = 1;
      for ( int i = 0 ; i < ndim ; i++ ) {
         d_cumdims[i+1] = d_cumdims[i] * dims.v[i];
         d_cumpads[i+1] = d_cumpads[i] * pads.v[i];
      }

   }
   __syncthreads();

   //
   // set up indices for main block
   //
   tid = threadIdx.x + threadIdx.y*blockDim.x + blockIdx.x*blockDim.y*blockDim.x + blockIdx.y*gridDim.x*blockDim.y*blockDim.x; // Thread ID in global scope - every thread solves one system
   
   int ind = 0;

   for ( int j = 0; j < solvedim; j++)
      ind += (( tid /  d_cumdims[j] ) % dims.v[j]) * d_cumpads[j];
   for ( int j = solvedim+1; j < ndim; j++)
      ind += (( tid / (d_cumdims[j] / dims.v[solvedim])) % dims.v[j]) * d_cumpads[j];


   int stride   = d_cumpads[solvedim];
   int sys_size = dims.v[solvedim];

   if( tid<sys_n ) {
      trid_strided_multidim_kernel<REAL, VECTOR, INC>(a, b, c, d, u, ind, stride, sys_size);
   }
}


/* Rumor has it that the GPU kernels can take along about 256 bytes of arguments
 * If we assume that includes a Program Counter pointer, that means that we have:
 * 8 bytes PC
 * 5*8 bytes (a, b, c, d, u)
 * 4 bytes (ndim)
 * 4 bytes (solvedim)
 * 4 bytes (sys_n)
 * ===
 * 60 bytes
 *
 * That leaves ~196B.
 *
 * We need to pass 8 bytes per supported dimension (4 dim, 4 pad)
 * For a MAXDIM of 8, that is 64 bytes, well within the range.
 * 
 * So, our API:
 * __host__ launchSolve(a,b,c,d,u, ndim, int*dims, int*pads, solvedim, sys_n)
 */
#if MAXDIM > 8
#error "Code needs updated to support DIMS > 8... Verify GPU can handle it"
#endif

template<typename REAL, typename VECTOR, int INC>
void trid_strided_multidim(const dim3 &grid, const dim3 & block,
                           const VECTOR* __restrict__ a,
                           const VECTOR* __restrict__ b,
                           const VECTOR* __restrict__ c,
                           VECTOR* __restrict__ d,
                           VECTOR* __restrict__ u, int ndim,
                           int solvedim, int sys_n,
                           const int* __restrict__ dims,
                           const int* __restrict__ pads) {

   int8 a_dims, a_pads;
   memcpy(&a_dims.v, dims, ndim*sizeof(int));
   memcpy(&a_pads.v, pads, ndim*sizeof(int));

   trid_strided_multidim<REAL, VECTOR, INC><<<grid, block>>>(a, b, c, d, u, ndim, solvedim, sys_n,
         a_dims, a_pads);
}


#endif
