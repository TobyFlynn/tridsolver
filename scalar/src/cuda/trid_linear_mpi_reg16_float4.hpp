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
// Based on trid_linear_reg16_float4.hpp written by Endre Laszlo, University of Oxford, endre.laszlo@oerc.ox.ac.uk, 2013-2014 

#ifndef TRID_LINEAR_MPI_REG16_FLOAT4_GPU_MPI__
#define TRID_LINEAR_MPI_REG16_FLOAT4_GPU_MPI__

#define VEC 8 
#define WARP_SIZE 32

#include <assert.h>
#include <sm_35_intrinsics.h>
#include <generics/generics/shfl.h>

#include "cuda_shfl.h"

typedef union {
  float4 vec[VEC/4];
  float  f[VEC];
} float16;

// transpose4x4xor() - exchanges data between 4 consecutive threads 
inline __device__ void transpose4x4xor(float16* la) {
  float4 tmp1;
  float4 tmp2;

  // Perform a 2-stage butterfly transpose

  // stage 1 (transpose each one of the 2x2 sub-blocks internally)
  if (threadIdx.x&1) {
    tmp1 = (*la).vec[0];
    tmp2 = (*la).vec[2];
  } else {
    tmp1 = (*la).vec[1];
    tmp2 = (*la).vec[3];
  }

  tmp1.x = trid_shfl_xor(tmp1.x,1);
  tmp1.y = trid_shfl_xor(tmp1.y,1);
  tmp1.z = trid_shfl_xor(tmp1.z,1);
  tmp1.w = trid_shfl_xor(tmp1.w,1);

  tmp2.x = trid_shfl_xor(tmp2.x,1);
  tmp2.y = trid_shfl_xor(tmp2.y,1);
  tmp2.z = trid_shfl_xor(tmp2.z,1);
  tmp2.w = trid_shfl_xor(tmp2.w,1);

  if (threadIdx.x&1) {
    (*la).vec[0] = tmp1;
    (*la).vec[2] = tmp2;
  } else {
    (*la).vec[1] = tmp1;
    (*la).vec[3] = tmp2;
  }

  // stage 2 (swap off-diagonal 2x2 blocks)
  if (threadIdx.x&2) {
    tmp1 = (*la).vec[0];
    tmp2 = (*la).vec[1];
  } else {
    tmp1 = (*la).vec[2];
    tmp2 = (*la).vec[3];
  }

  tmp1.x = trid_shfl_xor(tmp1.x,2);
  tmp1.y = trid_shfl_xor(tmp1.y,2);
  tmp1.z = trid_shfl_xor(tmp1.z,2);
  tmp1.w = trid_shfl_xor(tmp1.w,2);

  tmp2.x = trid_shfl_xor(tmp2.x,2);
  tmp2.y = trid_shfl_xor(tmp2.y,2);
  tmp2.z = trid_shfl_xor(tmp2.z,2);
  tmp2.w = trid_shfl_xor(tmp2.w,2);

  if (threadIdx.x&2) {
    (*la).vec[0] = tmp1;
    (*la).vec[1] = tmp2;
  } else {
    (*la).vec[2] = tmp1;
    (*la).vec[3] = tmp2;
  }
}

// ga - global array
// la - local array
inline __device__ void load_array_reg16(const float* __restrict__ ga, float16* la, int n, int woffset, int sys_pads) {
  int gind; // Global memory index of an element
  // Array indexing can be decided in compile time -> arrays will stay in registers
  // If trow and tcol are taken as an argument, they are not know in compile time -> no optimization
  int trow = (threadIdx.x % 32) / 4; // Threads' row index within a warp 
  int tcol =  threadIdx.x       % 4; // Threads' colum index within a warp 

  // Load 4 float4 values (64bytes) from an X-line
  gind = woffset + (4*(trow)) * sys_pads + tcol*4 + n; // First index in the X-line; woffset - warp offset in global memory
  int i;
  //#pragma unroll(4)
  for(i=0; i<4; i++) {
  //  (*la).vec[i] = *((float4*)&ga[gind]); 
    (*la).vec[i] = __ldg( ((float4*)&ga[gind]) ); 
    gind += sys_pads;
  }

  transpose4x4xor(la);
}

// Store a tile with 32x16 elements into 32 float16 struct allocated in registers. Every 4 consecutive threads cooperate to transpose and store a 4 x float4 sub-tile. 
// ga - global array
// la - local array
inline __device__ void store_array_reg16(float* __restrict__ ga, float16* la, int n, int woffset, int sys_pads) {
  int gind; // Global memory index of an element
  // Array indexing can be decided in compile time -> arrays will stay in registers
  // If trow and tcol are taken as an argument, they are not know in compile time -> no optimization
  int trow = (threadIdx.x % 32) / 4; // Threads' row index within a warp 
  int tcol =  threadIdx.x     % 4;   // Threads' colum index within a warp

  transpose4x4xor(la);

  gind = woffset + (4*(trow)) * sys_pads + tcol*4 + n;
  *((float4*)&ga[gind]) = (*la).vec[0];
  gind += sys_pads;
  *((float4*)&ga[gind]) = (*la).vec[1]; 
  gind += sys_pads;
  *((float4*)&ga[gind]) = (*la).vec[2]; 
  gind += sys_pads;
  *((float4*)&ga[gind]) = (*la).vec[3]; 
}

__global__ void
trid_linear_forward_float(const float *__restrict__ a, const float *__restrict__ b,
                    const float *__restrict__ c, const float *__restrict__ d,
                    float *__restrict__ aa, float *__restrict__ cc,
                    float *__restrict__ dd, float *__restrict__ boundaries,
                    int sys_size, int sys_pads, int sys_n) {
  // Thread ID in global scope - every thread solves one system
  const int tid = threadIdx.x + threadIdx.y * blockDim.x +
                  blockIdx.x * blockDim.y * blockDim.x +
                  blockIdx.y * gridDim.x * blockDim.y * blockDim.x;
  // Warp ID in global scope - the ID wich the thread belongs to
  const int wid = tid / WARP_SIZE;
  // Global memory offset: unique to a warp; 
  // every thread in a warp calculates the same woffset, which is the "begining" of 3D tile
  const int woffset = wid * WARP_SIZE * sys_pads;
  // These 4-threads do the regular memory read/write and data transpose
  const int optimized_solve = ((tid/4)*4+4 <= sys_n);
  // Among these 4-threads are some that have to be deactivated from global memory read/write
  const int boundary_solve  = !optimized_solve && ( tid < (sys_n) );
  // A thread is active only if it works on valid memory
  const int active_thread   = optimized_solve || boundary_solve;
  // Check if aligned memory 
  const int aligned         = (sys_pads % VEC) == 0;
  
  int n = 0;
  int b_ind = tid * 6;
  int ind = sys_pads * tid;
  
  float16 l_a, l_b, l_c, l_d, l_aa, l_cc, l_dd;
  float bb, a2, c2, d2;
  
  if(active_thread) {
    if(optimized_solve) {
      if(aligned) {
        // Process first vector separately
        load_array_reg16(a,&l_a,n, woffset, sys_size);
        load_array_reg16(b,&l_b,n, woffset, sys_size);
        load_array_reg16(c,&l_c,n, woffset, sys_size);
        load_array_reg16(d,&l_d,n, woffset, sys_size);
        
        for (int i = 0; i < 2; i++) {
          bb = 1.0f / l_b.f[i];
          d2 = bb * l_d.f[i];
          a2 = bb * l_a.f[i];
          c2 = bb * l_c.f[i];
          l_dd.f[i] = d2;
          l_aa.f[i] = a2;
          l_cc.f[i] = c2;
        }
        
        for(int i = 2; i < VEC; i++) {
          bb = 1.0f / (l_b.f[i] - l_a.f[i] * c2);
          d2 = (l_d.f[i] - l_a.f[i] * d2) * bb;
          a2 = (-l_a.f[i] * a2) * bb;
          c2 = l_c.f[i] * bb;
          l_dd.f[i] = d2;
          l_aa.f[i] = a2;
          l_cc.f[i] = c2;
        }
        
        store_array_reg16(dd,&l_dd,n, woffset, sys_size);
        store_array_reg16(cc,&l_cc,n, woffset, sys_size);
        store_array_reg16(aa,&l_aa,n, woffset, sys_size);
        
        // Forward pass
        for(n = VEC; n < sys_size; n += VEC) {
          load_array_reg16(a,&l_a,n, woffset, sys_size);
          load_array_reg16(b,&l_b,n, woffset, sys_size);
          load_array_reg16(c,&l_c,n, woffset, sys_size);
          load_array_reg16(d,&l_d,n, woffset, sys_size);
          #pragma unroll 16
          for(int i=0; i<VEC; i++) {
            bb = 1.0f / (l_b.f[i] - l_a.f[i] * c2);
            d2 = (l_d.f[i] - l_a.f[i] * d2) * bb;
            a2 = (-l_a.f[i] * a2) * bb;
            c2 = l_c.f[i] * bb;
            l_dd.f[i] = d2;
            l_aa.f[i] = a2;
            l_cc.f[i] = c2;
          }
          store_array_reg16(dd,&l_dd,n, woffset, sys_size);
          store_array_reg16(cc,&l_cc,n, woffset, sys_size);
          store_array_reg16(aa,&l_aa,n, woffset, sys_size);
        }
        
        n = sys_size - VEC;
        
        load_array_reg16(aa,&l_aa,n, woffset, sys_size);
        load_array_reg16(cc,&l_cc,n, woffset, sys_size);
        load_array_reg16(dd,&l_dd,n, woffset, sys_size);
        
        a2 = l_aa.f[VEC - 2];
        c2 = l_cc.f[VEC - 2];
        d2 = l_dd.f[VEC - 2];
        
        for(int i = VEC - 3; i >= 0; i--) {
          d2 = l_dd.f[i] - l_cc.f[i] * d2;
          a2 = l_aa.f[i] - l_cc.f[i] * a2;
          c2 = -l_cc.f[i] * c2;
          l_dd.f[i] = d2;
          l_cc.f[i] = c2;
          l_aa.f[i] = a2;
        }
        
        store_array_reg16(dd,&l_dd,n, woffset, sys_size);
        store_array_reg16(cc,&l_cc,n, woffset, sys_size);
        store_array_reg16(aa,&l_aa,n, woffset, sys_size);
        
        for(n = sys_size - (2*VEC); n > 0; n -= VEC) {
          load_array_reg16(aa,&l_aa,n, woffset, sys_size);
          load_array_reg16(cc,&l_cc,n, woffset, sys_size);
          load_array_reg16(dd,&l_dd,n, woffset, sys_size);
          
          for(int i = VEC - 1; i >= 0; i--) {
            d2 = l_dd.f[i] - l_cc.f[i] * d2;
            a2 = l_aa.f[i] - l_cc.f[i] * a2;
            c2 = -l_cc.f[i] * c2;
            l_dd.f[i] = d2;
            l_cc.f[i] = c2;
            l_aa.f[i] = a2;
          }
          
          store_array_reg16(dd,&l_dd,n, woffset, sys_size);
          store_array_reg16(cc,&l_cc,n, woffset, sys_size);
          store_array_reg16(aa,&l_aa,n, woffset, sys_size);
        }
        
        n = 0;
        
        load_array_reg16(aa,&l_aa,n, woffset, sys_size);
        load_array_reg16(cc,&l_cc,n, woffset, sys_size);
        load_array_reg16(dd,&l_dd,n, woffset, sys_size);
        
        for(int i = VEC - 1; i > 0; i--) {
          d2 = l_dd.f[i] - l_cc.f[i] * d2;
          a2 = l_aa.f[i] - l_cc.f[i] * a2;
          c2 = -l_cc.f[i] * c2;
          l_dd.f[i] = d2;
          l_cc.f[i] = c2;
          l_aa.f[i] = a2;
        }
        
        bb = 1.0f / (1.0f - l_cc.f[0] * a2);
        l_dd.f[0] = bb * (l_dd.f[0] - l_cc.f[0] * d2);
        l_aa.f[0] = bb * l_aa.f[0];
        l_cc.f[0] = bb * (-l_cc.f[0] * c2);
        
        store_array_reg16(dd,&l_dd,n, woffset, sys_size);
        store_array_reg16(cc,&l_cc,n, woffset, sys_size);
        store_array_reg16(aa,&l_aa,n, woffset, sys_size);
        
        // prepare boundaries for communication
        int i = tid * 6;
        boundaries[i + 0] = aa[ind];
        boundaries[i + 1] = aa[ind + sys_size - 1];
        boundaries[i + 2] = cc[ind];
        boundaries[i + 3] = cc[ind + sys_size - 1];
        boundaries[i + 4] = dd[ind];
        boundaries[i + 5] = dd[ind + sys_size - 1];
      } else {
        
        // Process first vector separately
        load_array_reg16(a,&l_a,n, woffset, sys_size);
        load_array_reg16(b,&l_b,n, woffset, sys_size);
        load_array_reg16(c,&l_c,n, woffset, sys_size);
        load_array_reg16(d,&l_d,n, woffset, sys_size);
        
        for (int i = 0; i < 2; i++) {
          bb = 1.0f / l_b.f[i];
          d2 = bb * l_d.f[i];
          a2 = bb * l_a.f[i];
          c2 = bb * l_c.f[i];
          l_dd.f[i] = d2;
          l_aa.f[i] = a2;
          l_cc.f[i] = c2;
        }
        
        for(int i = 2; i < VEC; i++) {
          bb = 1.0f / (l_b.f[i] - l_a.f[i] * c2);
          d2 = (l_d.f[i] - l_a.f[i] * d2) * bb;
          a2 = (-l_a.f[i] * a2) * bb;
          c2 = l_c.f[i] * bb;
          l_dd.f[i] = d2;
          l_aa.f[i] = a2;
          l_cc.f[i] = c2;
        }
        
        store_array_reg16(dd,&l_dd,n, woffset, sys_size);
        store_array_reg16(cc,&l_cc,n, woffset, sys_size);
        store_array_reg16(aa,&l_aa,n, woffset, sys_size);
        
        // Forward pass
        for(n = VEC; n < sys_size - VEC; n += VEC) {
          load_array_reg16(a,&l_a,n, woffset, sys_size);
          load_array_reg16(b,&l_b,n, woffset, sys_size);
          load_array_reg16(c,&l_c,n, woffset, sys_size);
          load_array_reg16(d,&l_d,n, woffset, sys_size);
          #pragma unroll 16
          for(int i=0; i<VEC; i++) {
            bb = 1.0f / (l_b.f[i] - l_a.f[i] * c2);
            d2 = (l_d.f[i] - l_a.f[i] * d2) * bb;
            a2 = (-l_a.f[i] * a2) * bb;
            c2 = l_c.f[i] * bb;
            l_dd.f[i] = d2;
            l_aa.f[i] = a2;
            l_cc.f[i] = c2;
          }
          store_array_reg16(dd,&l_dd,n, woffset, sys_size);
          store_array_reg16(cc,&l_cc,n, woffset, sys_size);
          store_array_reg16(aa,&l_aa,n, woffset, sys_size);
        }
        
        // Finish off unaligned part
        
        for(int i = n; i < sys_size; i++) {
          int loc_ind = ind + i;
          bb = 1.0f / (b[loc_ind] - a[loc_ind] * cc[loc_ind - 1]);
          dd[loc_ind] = (d[loc_ind] - a[loc_ind] * dd[loc_ind - 1]) * bb;
          aa[loc_ind] = (-a[loc_ind] * aa[loc_ind - 1]) * bb;
          cc[loc_ind] = c[loc_ind] * bb;
        }
        
        n -= VEC;
        
        a2 = aa[ind + sys_size - 2];
        c2 = cc[ind + sys_size - 2];
        d2 = dd[ind + sys_size - 2];
        
        for(int i = sys_size - 3; i >= n + VEC; i--) {
          int loc_ind = ind + i;
          d2 = dd[loc_ind] - cc[loc_ind] * d2;
          a2 = aa[loc_ind] - cc[loc_ind] * a2;
          c2 = -cc[loc_ind] * c2;
          dd[loc_ind] = d2;
          aa[loc_ind] = a2;
          cc[loc_ind] = c2;
        }
        
        for(; n > 0; n -= VEC) {
          load_array_reg16(aa,&l_aa,n, woffset, sys_size);
          load_array_reg16(cc,&l_cc,n, woffset, sys_size);
          load_array_reg16(dd,&l_dd,n, woffset, sys_size);
          
          for(int i = VEC - 1; i >= 0; i--) {
            d2 = l_dd.f[i] - l_cc.f[i] * d2;
            a2 = l_aa.f[i] - l_cc.f[i] * a2;
            c2 = -l_cc.f[i] * c2;
            l_dd.f[i] = d2;
            l_cc.f[i] = c2;
            l_aa.f[i] = a2;
          }
          
          store_array_reg16(dd,&l_dd,n, woffset, sys_size);
          store_array_reg16(cc,&l_cc,n, woffset, sys_size);
          store_array_reg16(aa,&l_aa,n, woffset, sys_size);
        }
        
        n = 0;
        
        load_array_reg16(aa,&l_aa,n, woffset, sys_size);
        load_array_reg16(cc,&l_cc,n, woffset, sys_size);
        load_array_reg16(dd,&l_dd,n, woffset, sys_size);
        
        for(int i = VEC - 1; i > 0; i--) {
          d2 = l_dd.f[i] - l_cc.f[i] * d2;
          a2 = l_aa.f[i] - l_cc.f[i] * a2;
          c2 = -l_cc.f[i] * c2;
          l_dd.f[i] = d2;
          l_cc.f[i] = c2;
          l_aa.f[i] = a2;
        }
        
        bb = 1.0f / (1.0f - l_cc.f[0] * a2);
        l_dd.f[0] = bb * (l_dd.f[0] - l_cc.f[0] * d2);
        l_aa.f[0] = bb * l_aa.f[0];
        l_cc.f[0] = bb * (-l_cc.f[0] * c2);
        
        store_array_reg16(dd,&l_dd,n, woffset, sys_size);
        store_array_reg16(cc,&l_cc,n, woffset, sys_size);
        store_array_reg16(aa,&l_aa,n, woffset, sys_size);
        
        // prepare boundaries for communication
        int i = tid * 6;
        boundaries[i + 0] = aa[ind];
        boundaries[i + 1] = aa[ind + sys_size - 1];
        boundaries[i + 2] = cc[ind];
        boundaries[i + 3] = cc[ind + sys_size - 1];
        boundaries[i + 4] = dd[ind];
        boundaries[i + 5] = dd[ind + sys_size - 1];
      }
    } else {
      //
      // forward pass
      //
      for (int i = 0; i < 2; ++i) {
        bb = 1.0f / b[ind + i];
        dd[ind + i] = bb * d[ind + i];
        aa[ind + i] = bb * a[ind + i];
        cc[ind + i] = bb * c[ind + i];
      }

      if (sys_size >= 3) {
        // eliminate lower off-diagonal
        for (int i = 2; i < sys_size; i++) {
          int loc_ind = ind + i;
          bb = 1.0f / (b[loc_ind] - a[loc_ind] * cc[loc_ind - 1]);
          dd[loc_ind] = (d[loc_ind] - a[loc_ind] * dd[loc_ind - 1]) * bb;
          aa[loc_ind] = (-a[loc_ind] * aa[loc_ind - 1]) * bb;
          cc[loc_ind] = c[loc_ind] * bb;
        }
        // Eliminate upper off-diagonal
        for (int i = sys_size - 3; i > 0; --i) {
          int loc_ind = ind + i;
          dd[loc_ind] = dd[loc_ind] - cc[loc_ind] * dd[loc_ind + 1];
          aa[loc_ind] = aa[loc_ind] - cc[loc_ind] * aa[loc_ind + 1];
          cc[loc_ind] = -cc[loc_ind] * cc[loc_ind + 1];
        }
        bb = 1.0f / (1.0f - cc[ind] * aa[ind + 1]);
        dd[ind] = bb * (dd[ind] - cc[ind] * dd[ind + 1]);
        aa[ind] = bb * aa[ind];
        cc[ind] = bb * (-cc[ind] * cc[ind + 1]);
      }
      // prepare boundaries for communication
      int i = tid * 6;
      boundaries[i + 0] = aa[ind];
      boundaries[i + 1] = aa[ind + sys_size - 1];
      boundaries[i + 2] = cc[ind];
      boundaries[i + 3] = cc[ind + sys_size - 1];
      boundaries[i + 4] = dd[ind];
      boundaries[i + 5] = dd[ind + sys_size - 1];
    }
  }
}

template <int INC>
__global__ void
trid_linear_backward_float(const float *__restrict__ aa, const float *__restrict__ cc,
                     const float *__restrict__ dd, float *__restrict__ d,
                     float *__restrict__ u, const float *__restrict__ boundaries,
                     int sys_size, int sys_pads, int sys_n) {
  // Thread ID in global scope - every thread solves one system
  const int tid = threadIdx.x + threadIdx.y * blockDim.x +
                  blockIdx.x * blockDim.y * blockDim.x +
                  blockIdx.y * gridDim.x * blockDim.y * blockDim.x;
  // Warp ID in global scope - the ID wich the thread belongs to
  const int wid = tid / WARP_SIZE;
  // Global memory offset: unique to a warp; 
  // every thread in a warp calculates the same woffset, which is the "begining" of 3D tile
  const int woffset = wid * WARP_SIZE * sys_pads;
  // These 4-threads do the regular memory read/write and data transpose
  const int optimized_solve = ((tid/4)*4+4 <= sys_n);
  // Among these 4-threads are some that have to be deactivated from global memory read/write
  const int boundary_solve  = !optimized_solve && ( tid < (sys_n) );
  // A thread is active only if it works on valid memory
  const int active_thread   = optimized_solve || boundary_solve;
  // Check if aligned memory 
  const int aligned         = (sys_pads % VEC) == 0;
  
  int n = 0;
  int ind = sys_pads * tid;
  
  float16 l_aa, l_cc, l_dd, l_d, l_u;
  
  if(active_thread) {
    double dd0 = boundaries[2 * tid];
    double ddn = boundaries[2 * tid + 1];
    if(optimized_solve) {
      if(aligned) {
        if(INC) {
          load_array_reg16(aa,&l_aa,n, woffset, sys_size);
          load_array_reg16(cc,&l_cc,n, woffset, sys_size);
          load_array_reg16(dd,&l_dd,n, woffset, sys_size);
          load_array_reg16(u,&l_u,n, woffset, sys_size);
          
          l_u.f[0] += dd0;
          
          for(int i = 1; i < VEC; i++) {
            l_u.f[i] += l_dd.f[i] - l_aa.f[i] * dd0 - l_cc.f[i] * ddn;
          }
          
          store_array_reg16(u,&l_u,n, woffset, sys_size);
          
          for(n = VEC; n < sys_size - VEC; n += VEC) {
            load_array_reg16(aa,&l_aa,n, woffset, sys_size);
            load_array_reg16(cc,&l_cc,n, woffset, sys_size);
            load_array_reg16(dd,&l_dd,n, woffset, sys_size);
            load_array_reg16(u,&l_u,n, woffset, sys_size);
            for(int i = 0; i < VEC; i++) {
              l_u.f[i] += l_dd.f[i] - l_aa.f[i] * dd0 - l_cc.f[i] * ddn;
            }
            store_array_reg16(u,&l_u,n, woffset, sys_size);
          }
          
          n = sys_size - VEC;
          load_array_reg16(aa,&l_aa,n, woffset, sys_size);
          load_array_reg16(cc,&l_cc,n, woffset, sys_size);
          load_array_reg16(dd,&l_dd,n, woffset, sys_size);
          load_array_reg16(u,&l_u,n, woffset, sys_size);
          
          for(int i = 0; i < VEC - 1; i++) {
            l_u.f[i] += l_dd.f[i] - l_aa.f[i] * dd0 - l_cc.f[i] * ddn;
          }
          
          l_u.f[VEC - 1] += ddn;
          
          store_array_reg16(u,&l_u,n, woffset, sys_size);
        } else {
          load_array_reg16(aa,&l_aa,n, woffset, sys_size);
          load_array_reg16(cc,&l_cc,n, woffset, sys_size);
          load_array_reg16(dd,&l_dd,n, woffset, sys_size);
          
          l_d.f[0] = dd0;
          
          for(int i = 1; i < VEC; i++) {
            l_d.f[i] = l_dd.f[i] - l_aa.f[i] * dd0 - l_cc.f[i] * ddn;
          }
          
          store_array_reg16(d,&l_d,n, woffset, sys_size);
          
          for(n = VEC; n < sys_size - VEC; n += VEC) {
            load_array_reg16(aa,&l_aa,n, woffset, sys_size);
            load_array_reg16(cc,&l_cc,n, woffset, sys_size);
            load_array_reg16(dd,&l_dd,n, woffset, sys_size);
            for(int i = 0; i < VEC; i++) {
              l_d.f[i] = l_dd.f[i] - l_aa.f[i] * dd0 - l_cc.f[i] * ddn;
            }
            store_array_reg16(d,&l_d,n, woffset, sys_size);
          }
          
          n = sys_size - VEC;
          load_array_reg16(aa,&l_aa,n, woffset, sys_size);
          load_array_reg16(cc,&l_cc,n, woffset, sys_size);
          load_array_reg16(dd,&l_dd,n, woffset, sys_size);
          
          for(int i = 0; i < VEC - 1; i++) {
            l_d.f[i] = l_dd.f[i] - l_aa.f[i] * dd0 - l_cc.f[i] * ddn;
          }
          
          l_d.f[VEC - 1] = ddn;
          
          store_array_reg16(d,&l_d,n, woffset, sys_size);
        }
      } else {
        if(INC) {
          load_array_reg16(aa,&l_aa,n, woffset, sys_size);
          load_array_reg16(cc,&l_cc,n, woffset, sys_size);
          load_array_reg16(dd,&l_dd,n, woffset, sys_size);
          load_array_reg16(u,&l_u,n, woffset, sys_size);
          
          l_u.f[0] += dd0;
          
          for(int i = 1; i < VEC; i++) {
            l_u.f[i] += l_dd.f[i] - l_aa.f[i] * dd0 - l_cc.f[i] * ddn;
          }
          
          store_array_reg16(u,&l_u,n, woffset, sys_size);
          
          for(n = VEC; n < sys_size - VEC; n += VEC) {
            load_array_reg16(aa,&l_aa,n, woffset, sys_size);
            load_array_reg16(cc,&l_cc,n, woffset, sys_size);
            load_array_reg16(dd,&l_dd,n, woffset, sys_size);
            load_array_reg16(u,&l_u,n, woffset, sys_size);
            for(int i = 0; i < VEC; i++) {
              l_u.f[i] += l_dd.f[i] - l_aa.f[i] * dd0 - l_cc.f[i] * ddn;
            }
            store_array_reg16(u,&l_u,n, woffset, sys_size);
          }
          
          for(; n < sys_size - 1; n++) {
            u[ind + n] += dd[ind + n] - aa[ind + n] * dd0 - cc[ind + n] * ddn;
          }
          
          u[ind + sys_size - 1] += ddn;
        } else {
          load_array_reg16(aa,&l_aa,n, woffset, sys_size);
          load_array_reg16(cc,&l_cc,n, woffset, sys_size);
          load_array_reg16(dd,&l_dd,n, woffset, sys_size);
          
          l_d.f[0] = dd0;
          
          for(int i = 1; i < VEC; i++) {
            l_d.f[i] = l_dd.f[i] - l_aa.f[i] * dd0 - l_cc.f[i] * ddn;
          }
          
          store_array_reg16(d,&l_d,n, woffset, sys_size);
          
          for(n = VEC; n < sys_size - VEC; n += VEC) {
            load_array_reg16(aa,&l_aa,n, woffset, sys_size);
            load_array_reg16(cc,&l_cc,n, woffset, sys_size);
            load_array_reg16(dd,&l_dd,n, woffset, sys_size);
            for(int i = 0; i < VEC; i++) {
              l_d.f[i] = l_dd.f[i] - l_aa.f[i] * dd0 - l_cc.f[i] * ddn;
            }
            store_array_reg16(d,&l_d,n, woffset, sys_size);
          }
          
          for(; n < sys_size - 1; n++) {
            d[ind + n] = dd[ind + n] - aa[ind + n] * dd0 - cc[ind + n] * ddn;
          }
          
          d[ind + sys_size - 1] = ddn;
        }
      }
    } else {
      if(INC) {
        u[ind] += dd0;
        
        for(int i = 1; i < sys_size - 1; i++) {
          u[ind + i] += dd[ind + i] - aa[ind + i] * dd0 - cc[ind + i] * ddn;
        }
        
        u[ind + sys_size - 1] += ddn;
      } else {
        d[ind] = dd0;
        
        for(int i = 1; i < sys_size - 1; i++) {
          d[ind + i] = dd[ind + i] - aa[ind + i] * dd0 - cc[ind + i] * ddn;
        }
        
        d[ind + sys_size - 1] = ddn;
      }
    }
  }
}

#endif
