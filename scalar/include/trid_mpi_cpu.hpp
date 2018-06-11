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

#ifndef __TRID_MPI_CPU_HPP
#define __TRID_MPI_CPU_HPP

#include "trid_simd.h"
#include "math.h"

#define N_MPI_MAX 128

//
// Thomas solver for reduced system
//
template<typename REAL>
inline void thomas_on_reduced(
    const REAL* __restrict__ aa_r, 
    const REAL* __restrict__ cc_r, 
          REAL* __restrict__ dd_r, 
    int N, 
    int stride) {
  int   i, ind = 0;
  REAL aa, bb, cc, dd, c2[N_MAX], d2[N_MAX];
  //
  // forward pass
  //
  bb    = static_cast<REAL>(1.0);
  cc    = cc_r[0];
  dd    = dd_r[0];
  c2[0] = cc;
  d2[0] = dd;

  for(i=1; i<N; i++) {
    ind   = ind + stride;
    aa    = aa_r[ind];
    bb    = static_cast<REAL>(1.0) - aa*cc;
    dd    = dd_r[ind] - aa*dd;
    bb    = static_cast<REAL>(1.0)/bb;
    cc    = bb*cc_r[ind];
    dd    = bb*dd;
    c2[i] = cc;
    d2[i] = dd;
  }
  //
  // reverse pass
  //
  dd_r[ind] = dd;
  for(i=N-2; i>=0; i--) {
    ind    = ind - stride;
    dd     = d2[i] - c2[i]*dd;
    dd_r[ind] = dd;
  }
}

//
// Modified Thomas forwards pass
//
// Each array should have a size of N, although the first element of a (a[0]) in
// the first process and the last element of c in the last process will not be
// used eventually
template<typename REAL>
inline void thomas_forward(
    const REAL *__restrict__ a, 
    const REAL *__restrict__ b, 
    const REAL *__restrict__ c, 
    const REAL *__restrict__ d, 
    const REAL *__restrict__ u, 
          REAL *__restrict__ aa, 
          REAL *__restrict__ cc, 
          REAL *__restrict__ dd, 
    int N, 
    int stride) {

  REAL bbi;

  if(N >=2) {
    // Start lower off-diagonal elimination
    for(int i=0; i<2; i++) {
      bbi   = static_cast<REAL>(1.0) / b[i * stride];
      dd[i] = d[i * stride] * bbi;
      aa[i] = a[i * stride] * bbi;
      cc[i] = c[i * stride] * bbi;
    }
    if(N >=3 ) {
      // Eliminate lower off-diagonal
      for(int i=2; i<N; i++) {
        bbi = static_cast<REAL>(1.0) /
              (b[i * stride] - a[i * stride] * cc[i - 1]);
        dd[i] = (d[i * stride] - a[i * stride] * dd[i - 1]) * bbi;
        aa[i] = (              - a[i * stride] * aa[i - 1]) * bbi;
        cc[i] =  c[i * stride]                              * bbi;
      }
      // Eliminate upper off-diagonal
      for(int i=N-3; i>0; i--) {
        dd[i] = dd[i] - cc[i]*dd[i+1];
        aa[i] = aa[i] - cc[i]*aa[i+1];
        cc[i] =       - cc[i]*cc[i+1];
      }
      bbi = static_cast<REAL>(1.0) / (static_cast<REAL>(1.0) - cc[0]*aa[1]);
      dd[0] =  bbi * ( dd[0] - cc[0]*dd[1] );
      aa[0] =  bbi *   aa[0];
      cc[0] =  bbi * (       - cc[0]*cc[1] );
    }
  }
  else {
    printf("One of the processes has fewer than 2 equations, this is not "
           "supported\n");
    exit(-1);
  }
}

//
// Modified Thomas backward pass
//
template<typename REAL>
inline void thomas_backward(
    const REAL *__restrict__ aa, 
    const REAL *__restrict__ cc, 
    const REAL *__restrict__ dd, 
          REAL *__restrict__ d, 
    int N, 
    int stride) {

  d[0] = dd[0];
  #pragma ivdep
  for (int i=1; i<N-1; i++) {
    d[i * stride] = dd[i] - aa[i]*dd[0] - cc[i]*dd[N-1];
  }
  d[(N-1) * stride] = dd[N-1];
}
#endif
