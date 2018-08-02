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

#ifndef __TRID_MPI_CPU_HPP
#define __TRID_MPI_CPU_HPP

#include "math.h"
#include "trid_simd.h"
#include <cassert>
#include <mpi.h>
#include <vector>

#define N_MPI_MAX 128

//
// Thomas solver for reduced system
//
template <typename REAL>
inline void thomas_on_reduced(const REAL *__restrict__ aa_r,
                              const REAL *__restrict__ cc_r,
                              REAL *__restrict__ dd_r, int N, int stride) {
  int i, ind = 0;
  REAL aa, bb, cc, dd, c2[N_MAX], d2[N_MAX];
  //
  // forward pass
  //
  bb = static_cast<REAL>(1.0);
  cc = cc_r[0];
  dd = dd_r[0];
  c2[0] = cc;
  d2[0] = dd;

  for (i = 1; i < N; i++) {
    ind = ind + stride;
    aa = aa_r[ind];
    bb = static_cast<REAL>(1.0) - aa * cc;
    dd = dd_r[ind] - aa * dd;
    bb = static_cast<REAL>(1.0) / bb;
    cc = bb * cc_r[ind];
    dd = bb * dd;
    c2[i] = cc;
    d2[i] = dd;
  }
  //
  // reverse pass
  //
  dd_r[ind] = dd;
  for (i = N - 2; i >= 0; i--) {
    ind = ind - stride;
    dd = d2[i] - c2[i] * dd;
    dd_r[ind] = dd;
  }
}

//
// Modified Thomas forwards pass
//
// Each array should have a size of N, although the first element of a (a[0]) in
// the first process and the last element of c in the last process will not be
// used eventually
template <typename REAL>
inline void
thomas_forward(const REAL *__restrict__ a, const REAL *__restrict__ b,
               const REAL *__restrict__ c, const REAL *__restrict__ d,
               const REAL *__restrict__ u, REAL *__restrict__ aa,
               REAL *__restrict__ cc, REAL *__restrict__ dd, int N,
               int stride) {

  REAL bbi;

  if (N >= 2) {
    // Start lower off-diagonal elimination
    for (int i = 0; i < 2; i++) {
      bbi = static_cast<REAL>(1.0) / b[i * stride];
      dd[i] = d[i * stride] * bbi;
      aa[i] = a[i * stride] * bbi;
      cc[i] = c[i * stride] * bbi;
    }
    if (N >= 3) {
      // Eliminate lower off-diagonal
      for (int i = 2; i < N; i++) {
        bbi = static_cast<REAL>(1.0) /
              (b[i * stride] - a[i * stride] * cc[i - 1]);
        dd[i] = (d[i * stride] - a[i * stride] * dd[i - 1]) * bbi;
        aa[i] = (-a[i * stride] * aa[i - 1]) * bbi;
        cc[i] = c[i * stride] * bbi;
      }
      // Eliminate upper off-diagonal
      for (int i = N - 3; i > 0; i--) {
        dd[i] = dd[i] - cc[i] * dd[i + 1];
        aa[i] = aa[i] - cc[i] * aa[i + 1];
        cc[i] = -cc[i] * cc[i + 1];
      }
      bbi = static_cast<REAL>(1.0) / (static_cast<REAL>(1.0) - cc[0] * aa[1]);
      dd[0] = bbi * (dd[0] - cc[0] * dd[1]);
      aa[0] = bbi * aa[0];
      cc[0] = bbi * (-cc[0] * cc[1]);
    }
  } else {
    printf("One of the processes has fewer than 2 equations, this is not "
           "supported\n");
    exit(-1);
  }
}

//
// Modified Thomas backward pass
//
template <typename REAL>
inline void thomas_backward(const REAL *__restrict__ aa,
                            const REAL *__restrict__ cc,
                            const REAL *__restrict__ dd, REAL *__restrict__ d,
                            int N, int stride) {

  d[0] = dd[0];
#pragma ivdep
  for (int i = 1; i < N - 1; i++) {
    d[i * stride] = dd[i] - aa[i] * dd[0] - cc[i] * dd[N - 1];
  }
  d[(N - 1) * stride] = dd[N - 1];
}

struct MpiSolverParams {
  // Communicator that includes every node calculating the same set of
  // equations as the current node.
  MPI_Comm communicator;
  // The size of the mesh; must be a `num_dims` large array.
  const int *size;
  // The number of dimensions in the mesh.
  int num_dims;
  // The index of the dimension along which the equations are defined.
  int equation_dim;
  // The index of the dimension along which the domain is decomposed.
  int domain_decomposition_dim;
  // The number of MPI processes.
  int num_mpi_procs;
  // The MPI rank of the current process.
  int mpi_rank;
  // The size of the local domain along the dimension of the equations.
  int local_equation_size;
};

//
// MPI solver
//
// Solves the tridiagonal linear equations on the mesh defined by `a`, `b`, `c`
// and `d`, and parameterized by `params`. The arrays are column-major (the
// consecutive dimension is the 0th one).
//
// Assumes that the domain is decomposed only in one dimension and this is the
// same as the dimension of the equations (i.e. `equation_dim ==
// domain_decomposition_dim`).
template <typename REAL>
inline void trid_solve_mpi(const MpiSolverParams &params, const REAL *a,
                           const REAL *b, const REAL *c, REAL *d) {
  assert(params.equation_dim == params.domain_decomposition_dim &&
         "trid_solve_mpi: only the case when the MPI decomposition dimension "
         "is the same as the dimension of the equation is supported");
  assert(params.equation_dim < params.num_dims);
  assert((
      (std::is_same<REAL, float>::value || std::is_same<REAL, double>::value) &&
      "trid_solve_mpi: only double or float values are supported"));

  int eq_stride = 1;
  for (size_t i = 0; i < params.equation_dim; ++i) {
    eq_stride *= params.size[i];
  }
  // The product of the sizes along the dimensions higher than solve_dim; needed
  // for the iteration later
  int outer_size = 1;
  for (size_t i = params.equation_dim + 1; i < params.num_dims; ++i) {
    outer_size *= params.size[i];
  }
  const size_t eq_size = params.size[params.equation_dim];
  // The start index of our domain along the dimension of the MPI
  // decomposition/solve_dim
  const size_t mpi_domain_offset =
      params.mpi_rank * (eq_size / params.num_mpi_procs);
  // The size of the equations / our domain
  const size_t local_eq_size = params.mpi_rank == params.num_mpi_procs - 1
                                   ? eq_size - mpi_domain_offset
                                   : eq_size / params.num_mpi_procs;

  const MPI_Datatype real_datatype =
      std::is_same<REAL, double>::value ? MPI_DOUBLE : MPI_FLOAT;

  // Local modifications to the coefficients
  std::vector<REAL> aa(params.local_equation_size),
      cc(params.local_equation_size), dd(params.local_equation_size);
  // MPI buffers (6 because 2 from each of the a, c and d coefficient arrays)
  std::vector<REAL> send_buf(6), receive_buf(6 * params.num_mpi_procs);
  // Reduced system
  std::vector<REAL> aa_r(2 * params.num_mpi_procs),
      cc_r(2 * params.num_mpi_procs), dd_r(2 * params.num_mpi_procs);

  // Calculation
  for (size_t outer_ind = 0; outer_ind < outer_size; ++outer_ind) {
    // Start of the domain for the slice defined by `outer_ind` in the global
    // mesh along the dimension of the decomposition. (Note: the arrays only
    // contain the local data.)
    const size_t domain_start = outer_ind * local_eq_size * eq_stride;
    for (size_t local_eq_start = 0; local_eq_start < eq_stride;
         ++local_eq_start) {
      const size_t equation_offset = domain_start + local_eq_start;
      thomas_forward<REAL>(a + equation_offset, b + equation_offset,
                           c + equation_offset, d + equation_offset, nullptr,
                           aa.data(), cc.data(), dd.data(),
                           params.local_equation_size, eq_stride);

      send_buf[0] = aa[0];
      send_buf[1] = aa[params.local_equation_size - 1];
      send_buf[2] = cc[0];
      send_buf[3] = cc[params.local_equation_size - 1];
      send_buf[4] = dd[0];
      send_buf[5] = dd[params.local_equation_size - 1];
      MPI_Allgather(send_buf.data(), 6, real_datatype, receive_buf.data(), 6,
                    real_datatype, params.communicator);
      for (int i = 0; i < params.num_mpi_procs; ++i) {
        aa_r[2 * i + 0] = receive_buf[6 * i + 0];
        aa_r[2 * i + 1] = receive_buf[6 * i + 1];
        cc_r[2 * i + 0] = receive_buf[6 * i + 2];
        cc_r[2 * i + 1] = receive_buf[6 * i + 3];
        dd_r[2 * i + 0] = receive_buf[6 * i + 4];
        dd_r[2 * i + 1] = receive_buf[6 * i + 5];
      }

      // indexing of cc_r, dd_r starts from 0
      // while indexing of aa_r starts from 1
      thomas_on_reduced<REAL>(aa_r.data(), cc_r.data(), dd_r.data(),
                              2 * params.num_mpi_procs, 1);

      dd[0] = dd_r[2 * params.mpi_rank];
      dd[params.local_equation_size - 1] = dd_r[2 * params.mpi_rank + 1];
      thomas_backward<REAL>(aa.data(), cc.data(), dd.data(),
                            d + equation_offset, params.local_equation_size,
                            eq_stride);
    }
  }
}

#endif
