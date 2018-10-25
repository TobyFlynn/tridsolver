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

#include "trid_cpu.h"
#include "transpose.hpp"
#include "trid_common.h"
#include "trid_simd.h"
#include <assert.h>
#include <stdio.h>
#include <cstring>
#define ROUND_DOWN(N, step) (((N) / (step)) * step)

#ifdef __MIC__ // Or #ifdef __KNC__ - more general option, future proof,
               // __INTEL_OFFLOAD is another option

__attribute__((target(mic))) inline void
load(SIMD_REG *__restrict__ dst, const FP *__restrict__ src, int n, int pad);

__attribute__((target(mic))) inline void
store(FP *__restrict__ dst, SIMD_REG *__restrict__ src, int n, int pad);

__attribute__((target(mic))) void
trid_x_transpose(const FP *__restrict a, const FP *__restrict b,
                 const FP *__restrict c, FP *__restrict d, FP *__restrict u,
                 int sys_size, int sys_pad, int stride);

__attribute__((target(mic))) void
trid_scalar_vec(const REAL *__restrict h_a, const REAL *__restrict h_b,
                const REAL *__restrict h_c, REAL *__restrict h_d,
                REAL *__restrict h_u, int N, int stride_a, int stride_b, int stride_c, int stride_d, int stride_u);

__attribute__((target(mic))) void
trid_scalar(const FP *__restrict a, const FP *__restrict b,
            const FP *__restrict c, FP *__restrict d, FP *__restrict u, int N,
            int stride_a, int stride_b, int stride_c, int stride_d, int stride_u);

#endif

inline void load(SIMD_REG *__restrict__ dst, const FP *__restrict__ src, int n,
                 int pad) {
  __assume_aligned(src, SIMD_WIDTH);
  __assume_aligned(dst, SIMD_WIDTH);
  for (int i = 0; i < SIMD_VEC; i++) {
    dst[i] = *(SIMD_REG *)&(src[i * pad + n]);
  }
}

inline void store(FP *__restrict__ dst, SIMD_REG *__restrict__ src, int n,
                  int pad) {
  __assume_aligned(src, SIMD_WIDTH);
  __assume_aligned(dst, SIMD_WIDTH);
  for (int i = 0; i < SIMD_VEC; i++) {
    *(SIMD_REG *)&(dst[i * pad + n]) = src[i];
  }
}

#ifdef __MIC__
#  if FPPREC == 0
#    define LOAD(reg, array, n, N)                                             \
      load(reg, array, n, N);                                                  \
      transpose16x16_intrinsic(reg);
#    define STORE(array, reg, n, N)                                            \
      transpose16x16_intrinsic(reg);                                           \
      store(array, reg, n, N);
#  elif FPPREC == 1
#    define LOAD(reg, array, n, N)                                             \
      load(reg, array, n, N);                                                  \
      transpose8x8_intrinsic(reg);
#    define STORE(array, reg, n, N)                                            \
      transpose8x8_intrinsic(reg);                                             \
      store(array, reg, n, N);
#  endif
#else
#  if FPPREC == 0
#    define LOAD(reg, array, n, N)                                             \
      load(reg, array, n, N);                                                  \
      transpose8x8_intrinsic(reg);
#    define STORE(array, reg, n, N)                                            \
      transpose8x8_intrinsic(reg);                                             \
      store(array, reg, n, N);
#  elif FPPREC == 1
#    define LOAD(reg, array, n, N)                                             \
      load(reg, array, n, N);                                                  \
      transpose4x4_intrinsic(reg);
#    define STORE(array, reg, n, N)                                            \
      transpose4x4_intrinsic(reg);                                             \
      store(array, reg, n, N);
#  endif
#endif

//
// tridiagonal-x solver; vectorised solution where the system dimension is the
// same as the vectorisation dimension
//
template <int inc>
void trid_x_transpose(const FP *__restrict a, const FP *__restrict b,
                      const FP *__restrict c, FP *__restrict d,
                      FP *__restrict u, int sys_size, int sys_pad, int stride) {

  __assume_aligned(a, SIMD_WIDTH);
  __assume_aligned(b, SIMD_WIDTH);
  __assume_aligned(c, SIMD_WIDTH);
  __assume_aligned(d, SIMD_WIDTH);

  assert((((long long)a) % SIMD_WIDTH) == 0);

  int i, ind = 0;
  SIMD_REG aa;
  SIMD_REG bb;
  SIMD_REG cc;
  SIMD_REG dd;

  SIMD_REG tmp1;
  SIMD_REG tmp2;

  SIMD_REG a_reg[SIMD_VEC];
  SIMD_REG b_reg[SIMD_VEC];
  SIMD_REG c_reg[SIMD_VEC];
  SIMD_REG d_reg[SIMD_VEC];
  SIMD_REG u_reg[SIMD_VEC];

  SIMD_REG tmp_reg[SIMD_VEC];

  SIMD_REG c2[N_MAX];
  SIMD_REG d2[N_MAX];

  //
  // forward pass
  //
  int n = 0;
  SIMD_REG ones = SIMD_SET1_P(1.0F);

  LOAD(a_reg, a, n, sys_pad);
  LOAD(b_reg, b, n, sys_pad);
  LOAD(c_reg, c, n, sys_pad);
  LOAD(d_reg, d, n, sys_pad);

  bb = b_reg[0];
#if FPPREC == 0
  bb = SIMD_RCP_P(bb);
#elif FPPREC == 1
  bb = SIMD_DIV_P(ones, bb);
#endif
  cc = c_reg[0];
  cc = SIMD_MUL_P(bb, cc);
  dd = d_reg[0];
  dd = SIMD_MUL_P(bb, dd);
  c2[0] = cc;
  d2[0] = dd;

  for (i = 1; i < SIMD_VEC; i++) {
    aa = a_reg[i];
#ifdef __MIC__
    bb = SIMD_FNMADD_P(aa, cc, b_reg[i]);
    dd = SIMD_FNMADD_P(aa, dd, d_reg[i]);
#else
    bb = SIMD_SUB_P(b_reg[i], SIMD_MUL_P(aa, cc));
    dd = SIMD_SUB_P(d_reg[i], SIMD_MUL_P(aa, dd));
#endif
#if FPPREC == 0
    bb = SIMD_RCP_P(bb);
#elif FPPREC == 1
    bb = SIMD_DIV_P(ones, bb);
#endif
    cc = SIMD_MUL_P(bb, c_reg[i]);
    dd = SIMD_MUL_P(bb, dd);
    c2[n + i] = cc;
    d2[n + i] = dd;
  }

  for (n = SIMD_VEC; n < (sys_size / SIMD_VEC) * SIMD_VEC; n += SIMD_VEC) {
    LOAD(a_reg, a, n, sys_pad);
    LOAD(b_reg, b, n, sys_pad);
    LOAD(c_reg, c, n, sys_pad);
    LOAD(d_reg, d, n, sys_pad);
    for (i = 0; i < SIMD_VEC; i++) {
      aa = a_reg[i];
#ifdef __MIC__
      bb = SIMD_FNMADD_P(aa, cc, b_reg[i]);
      dd = SIMD_FNMADD_P(aa, dd, d_reg[i]);
#else
      bb = SIMD_SUB_P(b_reg[i], SIMD_MUL_P(aa, cc));
      dd = SIMD_SUB_P(d_reg[i], SIMD_MUL_P(aa, dd));
#endif
#if FPPREC == 0
      bb = SIMD_RCP_P(bb);
#elif FPPREC == 1
      bb = SIMD_DIV_P(ones, bb);
#endif
      cc = SIMD_MUL_P(bb, c_reg[i]);
      dd = SIMD_MUL_P(bb, dd);
      c2[n + i] = cc;
      d2[n + i] = dd;
    }
  }

  if (sys_size != sys_pad) {
    n = (sys_size / SIMD_VEC) * SIMD_VEC;
    LOAD(a_reg, a, n, sys_pad);
    LOAD(b_reg, b, n, sys_pad);
    LOAD(c_reg, c, n, sys_pad);
    LOAD(d_reg, d, n, sys_pad);
    for (i = 0; (n + i) < sys_size; i++) {
      aa = a_reg[i];
#ifdef __MIC__
      bb = SIMD_FNMADD_P(aa, cc, b_reg[i]);
      dd = SIMD_FNMADD_P(aa, dd, d_reg[i]);
#else
      bb = SIMD_SUB_P(b_reg[i], SIMD_MUL_P(aa, cc));
      dd = SIMD_SUB_P(d_reg[i], SIMD_MUL_P(aa, dd));
#endif
#if FPPREC == 0
      bb = SIMD_RCP_P(bb);
#elif FPPREC == 1
      bb = SIMD_DIV_P(ones, bb);
#endif
      cc = SIMD_MUL_P(bb, c_reg[i]);
      dd = SIMD_MUL_P(bb, dd);
      c2[n + i] = cc;
      d2[n + i] = dd;
    }
    d_reg[i - 1] = dd;
    for (i = i - 2; i >= 0; i--) {
      dd = SIMD_SUB_P(d2[n + i], SIMD_MUL_P(c2[n + i], dd));
      d_reg[i] = dd;
    }
    if (inc) {
      LOAD(u_reg, u, n, sys_pad);
      for (int j = 0; j < SIMD_VEC; j++)
        u_reg[j] = SIMD_ADD_P(u_reg[j], d_reg[j]);
      STORE(u, u_reg, n, sys_pad);
    } else
      STORE(d, d_reg, n, sys_pad);
  } else {

    //
    // reverse pass
    //
    d_reg[SIMD_VEC - 1] = dd;
    n -= SIMD_VEC;
    for (i = SIMD_VEC - 2; i >= 0; i--) {
      dd = SIMD_SUB_P(d2[n + i], SIMD_MUL_P(c2[n + i], dd));
      d_reg[i] = dd;
    }
    if (inc) {
      LOAD(u_reg, u, n, sys_pad);
      for (int j = 0; j < SIMD_VEC; j++)
        u_reg[j] = SIMD_ADD_P(u_reg[j], d_reg[j]);
      STORE(u, u_reg, n, sys_pad);
    } else
      STORE(d, d_reg, n, sys_pad);
  }

  for (n = (sys_size / SIMD_VEC) * SIMD_VEC - SIMD_VEC; n >= 0; n -= SIMD_VEC) {
    for (i = (SIMD_VEC - 1); i >= 0; i--) {
      dd = SIMD_SUB_P(d2[n + i], SIMD_MUL_P(c2[n + i], dd));
      d_reg[i] = dd;
    }
    if (inc) {
      LOAD(u_reg, u, n, sys_pad);
      for (int j = 0; j < SIMD_VEC; j++)
        u_reg[j] = SIMD_ADD_P(u_reg[j], d_reg[j]);
      STORE(u, u_reg, n, sys_pad);
    } else
      STORE(d, d_reg, n, sys_pad);
  }
}

//
// tridiagonal solver; vectorised solution where the system dimension is not
// the same as the vectorisation dimension
//
template <typename REAL, typename VECTOR, int INC>
void trid_scalar_vec(const REAL *__restrict h_a, const REAL *__restrict h_b,
                     const REAL *__restrict h_c, REAL *__restrict h_d,
                     REAL *__restrict h_u, int N, int stride_a, int stride_b, int stride_c, int stride_d, int stride_u) {

  int i;
  int ind_a = 0;
  int ind_b = 0;
  int ind_c = 0;
  int ind_d = 0;
  int ind_u = 0;
  VECTOR aa, bb, cc, dd, c2[N_MAX], d2[N_MAX];

  VECTOR *__restrict a = (VECTOR *)h_a;
  VECTOR *__restrict b = (VECTOR *)h_b;
  VECTOR *__restrict c = (VECTOR *)h_c;
  VECTOR *__restrict d = (VECTOR *)h_d;
  VECTOR *__restrict u = (VECTOR *)h_u;

  VECTOR SIMD_CONSTRUCTOR(ones, 1.0f);

  //
  // forward pass
  //
  bb = ones / b[0];
  cc = bb * c[0];
  dd = bb * d[0];
  c2[0] = cc;
  d2[0] = dd;

  for (i = 1; i < N; i++) {
    ind_a = ind_a + stride_a;
    ind_b = ind_b + stride_b;
    ind_c = ind_c + stride_c;
    ind_d = ind_d + stride_d;
    ind_u = ind_u + stride_u;
    aa = a[ind_a];
    bb = b[ind_b] - aa * cc;
    dd = d[ind_d] - aa * dd;
    bb = ones / bb;
    cc = bb * c[ind_c];
    dd = bb * dd;
    c2[i] = cc;
    d2[i] = dd;
  }
  //
  // reverse pass
  //
  if (INC)
    u[ind_u] += dd;
  else
    d[ind_d] = dd;
  for (i = N - 2; i >= 0; i--) {
    ind_u = ind_u - stride_u;
    ind_d = ind_d - stride_d;
    dd = d2[i] - c2[i] * dd;
    if (INC)
      u[ind_u] += dd;
    else
      d[ind_d] = dd;
  }
}

//
// tridiagonal solver; simple non-vectorised solution
//
template <int INC>
void trid_scalar(const FP *__restrict a, const FP *__restrict b,
                 const FP *__restrict c, FP *__restrict d, FP *__restrict u,
                 int N, int stride_a, int stride_b, int stride_c, int stride_d, int stride_u) {
  int i;
  int ind_a = 0;
  int ind_b = 0;
  int ind_c = 0;
  int ind_d = 0;
  int ind_u = 0;
  FP aa, bb, cc, dd, c2[N_MAX], d2[N_MAX];
  //
  // forward pass
  //
  bb = 1.0F / b[0];
  cc = bb * c[0];
  dd = bb * d[0];
  c2[0] = cc;
  d2[0] = dd;

  for (i = 1; i < N; i++) {
    ind_a = ind_a + stride_a;
    ind_b = ind_b + stride_b;
    ind_c = ind_c + stride_c;
    ind_d = ind_d + stride_d;
    ind_u = ind_u + stride_u;
    aa = a[ind_a];
    bb = b[ind_b] - aa * cc;
    dd = d[ind_d] - aa * dd;
    bb = 1.0F / bb;
    cc = bb * c[ind_c];
    dd = bb * dd;
    c2[i] = cc;
    d2[i] = dd;
  }
  //
  // reverse pass
  //
  if (INC)
    u[ind_u] += dd;
  else
    d[ind_d] = dd;
  for (i = N - 2; i >= 0; i--) {
    ind_d = ind_d - stride_d;
    ind_u = ind_u - stride_u;
    dd = d2[i] - c2[i] * dd;
    if (INC)
      u[ind_u] += dd;
    else
      d[ind_d] = dd;
  }
}

//
// Function for selecting the proper setup for solve in a specific dimension
//
void tridMultiDimBatchSolve(const FP *a, const int *a_pads, const FP *b, const int *b_pads, const FP *c, const int *c_pads, FP *d, const int *d_pads, FP *u, const int *u_pads,
                            int ndim, int solvedim, int *dims,
                            int inc) {
 
  if (solvedim == 0) {
    int sys_stride = 1; // Stride between the consecutive elements of a system
    int sys_size = dims[0]; // Size (length) of a system
    int sys_pads = a_pads[0]; // Padded sizes along each ndim number of dimensions
    int sys_n_lin =
        dims[1] * dims[2]; // = cumdims[solve] // Number of systems to be solved
    bool equal_pads = !(
            memcmp(a_pads, b_pads, ndim*sizeof(int)) ||
            memcmp(a_pads, c_pads, ndim*sizeof(int)) ||
            memcmp(a_pads, d_pads, ndim*sizeof(int)) ||
            memcmp(a_pads, u_pads, ndim*sizeof(int)) );

    // FIXME fix and re-enable vectorisation in x-dim
    if (equal_pads && (sys_pads % SIMD_VEC) == 0  && ((long)a & 0x3F)==0 && ((long)b & 0x3F)==0 && ((long)c & 0x3F)==0 && ((long)d & 0x3F)==0 && ((long)u & 0x3F)==0) {
#pragma omp parallel for collapse(2)
      for (int k = 0; k < dims[2]; k++) {
        for (int j = 0; j < ROUND_DOWN(dims[1], SIMD_VEC); j += SIMD_VEC) {
          int ind_a = k * a_pads[0] * a_pads[1] + j * a_pads[0];
          int ind_b = k * b_pads[0] * b_pads[1] + j * b_pads[0];
          int ind_c = k * c_pads[0] * c_pads[1] + j * c_pads[0];
          int ind_d = k * d_pads[0] * d_pads[1] + j * d_pads[0];
          int ind_u = k * u_pads[0] * u_pads[1] + j * u_pads[0];
          if (inc)
            trid_x_transpose<1>(&a[ind_a], &b[ind_b], &c[ind_c], &d[ind_d], &u[ind_u],
                                sys_size, sys_pads, sys_stride);
          else
            trid_x_transpose<0>(&a[ind_a], &b[ind_b], &c[ind_c], &d[ind_d], &u[ind_u],
                                sys_size, sys_pads, sys_stride);
        }
      }
      if (ROUND_DOWN(dims[1], SIMD_VEC) <
          dims[1]) { // If there is leftover, fork threads an compute it
#pragma omp parallel for collapse(2)
        for (int k = 0; k < dims[2]; k++) {
          for (int j = ROUND_DOWN(dims[1], SIMD_VEC); j < dims[1]; j++) {
              int ind_a = k * a_pads[0] * a_pads[1] + j * a_pads[0];
              int ind_b = k * b_pads[0] * b_pads[1] + j * b_pads[0];
              int ind_c = k * c_pads[0] * c_pads[1] + j * c_pads[0];
              int ind_d = k * d_pads[0] * d_pads[1] + j * d_pads[0];
              int ind_u = k * u_pads[0] * u_pads[1] + j * u_pads[0];
            if (inc)
              trid_scalar<1>(&a[ind_a], &b[ind_b], &c[ind_c], &d[ind_d], &u[ind_u],
                             sys_size, 1,1,1,1,1);
            else
              trid_scalar<0>(&a[ind_a], &b[ind_b], &c[ind_c], &d[ind_d], &u[ind_u],
                             sys_size, 1,1,1,1,1);
          }
        }
      }
    } else {
#pragma omp parallel for collapse(2)
      for (int k = 0; k < dims[2]; k++) {
        for (int j = 0; j < dims[1]; j++) {
          int ind_a = k * a_pads[0] * a_pads[1] + j * a_pads[0];
          int ind_b = k * b_pads[0] * b_pads[1] + j * b_pads[0];
          int ind_c = k * c_pads[0] * c_pads[1] + j * c_pads[0];
          int ind_d = k * d_pads[0] * d_pads[1] + j * d_pads[0];
          int ind_u = k * u_pads[0] * u_pads[1] + j * u_pads[0];
          if (inc)
            trid_scalar<1>(&a[ind_a], &b[ind_b], &c[ind_c], &d[ind_d], &u[ind_u],
                           sys_size, 1,1,1,1,1);
          else
            trid_scalar<0>(&a[ind_a], &b[ind_b], &c[ind_c], &d[ind_d], &u[ind_u],
                           sys_size, 1,1,1,1,1);
        }
      }
    }
  } else if (solvedim == 1) {
    int sys_size = dims[1]; // Size (length) of a system

#pragma omp parallel for collapse(2)
    for (int k = 0; k < dims[2]; k++) {
      for (int i = 0; i < ROUND_DOWN(dims[0], SIMD_VEC); i += SIMD_VEC) {
          int ind_a = k * a_pads[0] * a_pads[1] + i;
          int ind_b = k * b_pads[0] * b_pads[1] + i;
          int ind_c = k * c_pads[0] * c_pads[1] + i;
          int ind_d = k * d_pads[0] * d_pads[1] + i;
          int ind_u = k * u_pads[0] * u_pads[1] + i;
        if (inc)
          trid_scalar_vec<FP, VECTOR, 1>(&a[ind_a], &b[ind_b], &c[ind_c], &d[ind_d],
                                         &u[ind_u], sys_size,
                                         a_pads[0] / SIMD_VEC,
                                         b_pads[0] / SIMD_VEC,
                                         c_pads[0] / SIMD_VEC,
                                         d_pads[0] / SIMD_VEC,
                                         u_pads[0] / SIMD_VEC);
        else
          trid_scalar_vec<FP, VECTOR, 0>(&a[ind_a], &b[ind_b], &c[ind_c], &d[ind_d],
                                         &u[ind_u], sys_size,
                                         a_pads[0] / SIMD_VEC,
                                         b_pads[0] / SIMD_VEC,
                                         c_pads[0] / SIMD_VEC,
                                         d_pads[0] / SIMD_VEC,
                                         u_pads[0] / SIMD_VEC);
      }
    }
    if (ROUND_DOWN(dims[0], SIMD_VEC) <
        dims[0]) { // If there is leftover, fork threads an compute it
#pragma omp parallel for collapse(2)
      for (int k = 0; k < dims[2]; k++) {
        for (int i = ROUND_DOWN(dims[0], SIMD_VEC); i < dims[0]; i++) {
          int ind_a = k * a_pads[0] * a_pads[1] + i;
          int ind_b = k * b_pads[0] * b_pads[1] + i;
          int ind_c = k * c_pads[0] * c_pads[1] + i;
          int ind_d = k * d_pads[0] * d_pads[1] + i;
          int ind_u = k * u_pads[0] * u_pads[1] + i;
          if (inc)
            trid_scalar<1>(&a[ind_a], &b[ind_b], &c[ind_c], &d[ind_d], &u[ind_u],
                           sys_size, 
                                         a_pads[0],
                                         b_pads[0],
                                         c_pads[0],
                                         d_pads[0],
                                         u_pads[0]);
          else
            trid_scalar<0>(&a[ind_a], &b[ind_b], &c[ind_c], &d[ind_d], &u[ind_u],
                           sys_size,
                                         a_pads[0],
                                         b_pads[0],
                                         c_pads[0],
                                         d_pads[0],
                                         u_pads[0]);
        }
      }
    }
  } else if (solvedim == 2) {
    int sys_size = dims[2]; // Size (length) of a system

#pragma omp parallel for collapse(2) // Interleaved scheduling for better data
                                     // locality and thus lower TLB miss rate
    for (int j = 0; j < dims[1]; j++) {
      for (int i = 0; i < ROUND_DOWN(dims[0], SIMD_VEC); i += SIMD_VEC) {
          int ind_a = j * a_pads[0] + i;
          int ind_b = j * b_pads[0] + i;
          int ind_c = j * c_pads[0] + i;
          int ind_d = j * d_pads[0] + i;
          int ind_u = j * u_pads[0] + i;
        if (inc)
          trid_scalar_vec<FP, VECTOR, 1>(&a[ind_a], &b[ind_b], &c[ind_c], &d[ind_d],
                                         &u[ind_u], sys_size,
                                         a_pads[0] * a_pads[1] / SIMD_VEC,
                                         b_pads[0] * b_pads[1] / SIMD_VEC,
                                         c_pads[0] * c_pads[1] / SIMD_VEC,
                                         d_pads[0] * d_pads[1] / SIMD_VEC,
                                         u_pads[0] * u_pads[1] / SIMD_VEC);
        else
          trid_scalar_vec<FP, VECTOR, 0>(&a[ind_a], &b[ind_b], &c[ind_c], &d[ind_d],
                                         &u[ind_u], sys_size,
                                         a_pads[0] * a_pads[1] / SIMD_VEC,
                                         b_pads[0] * b_pads[1] / SIMD_VEC,
                                         c_pads[0] * c_pads[1] / SIMD_VEC,
                                         d_pads[0] * d_pads[1] / SIMD_VEC,
                                         u_pads[0] * u_pads[1] / SIMD_VEC);
      }
    }
    if (ROUND_DOWN(dims[0], SIMD_VEC) <
        dims[0]) { // If there is leftover, fork threads an compute it
#pragma omp parallel for collapse(2)
      for (int j = 0; j < dims[1]; j++) {
        for (int i = ROUND_DOWN(dims[0], SIMD_VEC); i < dims[0]; i++) {
          int ind_a = j * a_pads[0] + i;
          int ind_b = j * b_pads[0] + i;
          int ind_c = j * c_pads[0] + i;
          int ind_d = j * d_pads[0] + i;
          int ind_u = j * u_pads[0] + i;
          if (inc)
            trid_scalar<1>(&a[ind_a], &b[ind_b], &c[ind_c], &d[ind_d], &u[ind_u],
                           sys_size, 
                                         a_pads[0] * a_pads[1],
                                         b_pads[0] * b_pads[1],
                                         c_pads[0] * c_pads[1],
                                         d_pads[0] * d_pads[1],
                                         u_pads[0] * u_pads[1]);
          else
            trid_scalar<0>(&a[ind_a], &b[ind_b], &c[ind_c], &d[ind_d], &u[ind_u],
                           sys_size,
                                         a_pads[0] * a_pads[1],
                                         b_pads[0] * b_pads[1],
                                         c_pads[0] * c_pads[1],
                                         d_pads[0] * d_pads[1],
                                         u_pads[0] * u_pads[1]);
        }
      }
    }
  }
}

#if FPPREC == 0


tridStatus_t tridSmtsvStridedBatchPadded(const float *a, const int *a_pad, const float *b, const int *b_pad,
                                   const float *c, const int *c_pad, float *d, const int *d_pad,
                                   int ndim, int solvedim, int *dims) {

  tridMultiDimBatchSolve(a, a_pad, b, b_pad, c, c_pad, d, d_pad, NULL, a_pad, ndim, solvedim, dims, 0);
  return TRID_STATUS_SUCCESS;
}

tridStatus_t tridSmtsvStridedBatch(const float *a, const float *b,
                                   const float *c, float *d, float *u, int ndim,
                                   int solvedim, int *dims, int *pads) {
  tridMultiDimBatchSolve(a, pads, b, pads, c, pads, d, pads, NULL, NULL, ndim, solvedim, dims, 0);
  return TRID_STATUS_SUCCESS;
}


tridStatus_t tridSmtsvStridedBatchPaddedInc(const float *a, const int *a_pad, const float *b, const int *b_pad,
                                   const float *c, const int *c_pad, float *d, const int *d_pad, float *u, const int *u_pad,
                                   int ndim, int solvedim, int *dims) {
  tridMultiDimBatchSolve(a, a_pad, b, b_pad, c, c_pad, d, d_pad, u, u_pad, ndim, solvedim, dims, 1);
  return TRID_STATUS_SUCCESS;
}

tridStatus_t tridSmtsvStridedBatchInc(const float *a, const float *b,
                                      const float *c, float *d, float *u,
                                      int ndim, int solvedim, int *dims,
                                      int *pads) {
  tridMultiDimBatchSolve(a, pads, b, pads, c, pads, d, pads, u, pads, ndim, solvedim, dims, 1);
  return TRID_STATUS_SUCCESS;
}


void trid_scalarS(float *__restrict a, float *__restrict b, float *__restrict c,
                  float *__restrict d, float *__restrict u, int N, int stride) {

  trid_scalar<0>(a, b, c, d, u, N, stride, stride, stride, stride, stride);
}


void trid_x_transposeS(float *__restrict a, float *__restrict b,
                       float *__restrict c, float *__restrict d,
                       float *__restrict u, int sys_size, int sys_pad,
                       int stride) {

  trid_x_transpose<0>(a, b, c, d, u, sys_size, sys_pad, stride);
}


void trid_scalar_vecS(const float *__restrict a, const float *__restrict b,
                      const float *__restrict c, float *__restrict d,
                      float *__restrict u, int N, int stride) {

  trid_scalar_vec<FP, VECTOR, 0>(a, b, c, d, u, N, stride, stride, stride, stride, stride);
}


void trid_scalar_vecSInc(float *__restrict a, float *__restrict b,
                         float *__restrict c, float *__restrict d,
                         float *__restrict u, int N, int stride) {

  trid_scalar_vec<FP, VECTOR, 1>(a, b, c, d, u, N, stride, stride, stride, stride, stride);
}

#elif FPPREC == 1


tridStatus_t tridDmtsvStridedBatchPadded(const double *a, const int *a_pad, const double *b, const int *b_pad,
                                   const double *c, const int *c_pad, double *d, const int *d_pad,
                                   int ndim, int solvedim, int *dims) {
  tridMultiDimBatchSolve(a, a_pad, b, b_pad, c, c_pad, d, d_pad, NULL, a_pad, ndim, solvedim, dims, 0);
  return TRID_STATUS_SUCCESS;
}

tridStatus_t tridDmtsvStridedBatch(const double *a, const double *b,
                                   const double *c, double *d, double *u,
                                   int ndim, int solvedim, int *dims,
                                   int *pads) {
  tridMultiDimBatchSolve(a, pads, b, pads, c, pads, d, pads, NULL, pads, ndim, solvedim, dims, 0);
  return TRID_STATUS_SUCCESS;
}


tridStatus_t tridDmtsvStridedBatchPaddedInc(const double *a, const int *a_pad, const double *b, const int *b_pad,
                                   const double *c, const int *c_pad, double *d, const int *d_pad, double *u, const int *u_pad,
                                   int ndim, int solvedim, int *dims) {
  tridMultiDimBatchSolve(a, a_pad, b, b_pad, c, c_pad, d, d_pad, u, u_pad, ndim, solvedim, dims, 1);
  return TRID_STATUS_SUCCESS;
}

tridStatus_t tridDmtsvStridedBatchInc(const double *a, const double *b,
                                      const double *c, double *d, double *u,
                                      int ndim, int solvedim, int *dims,
                                      int *pads) {
  tridMultiDimBatchSolve(a, pads, b, pads, c, pads, d, pads, u, pads, ndim, solvedim, dims, 1);
  return TRID_STATUS_SUCCESS;
}


void trid_scalarD(double *__restrict a, double *__restrict b,
                  double *__restrict c, double *__restrict d,
                  double *__restrict u, int N, int stride) {

  trid_scalar<0>(a, b, c, d, u, N, stride, stride, stride, stride, stride);
}


void trid_x_transposeD(double *__restrict a, double *__restrict b,
                       double *__restrict c, double *__restrict d,
                       double *__restrict u, int sys_size, int sys_pad,
                       int stride) {

  trid_x_transpose<0>(a, b, c, d, u, sys_size, sys_pad, stride);
}


void trid_scalar_vecD(double *__restrict a, double *__restrict b,
                      double *__restrict c, double *__restrict d,
                      double *__restrict u, int N, int stride) {

  trid_scalar_vec<FP, VECTOR, 0>(a, b, c, d, u, N, stride, stride, stride, stride, stride);
}


void trid_scalar_vecDInc(double *__restrict a, double *__restrict b,
                         double *__restrict c, double *__restrict d,
                         double *__restrict u, int N, int stride) {

  trid_scalar_vec<FP, VECTOR, 1>(a, b, c, d, u, N, stride, stride, stride, stride, stride);
}
#endif
