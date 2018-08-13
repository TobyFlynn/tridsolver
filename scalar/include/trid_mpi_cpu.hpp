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
  // The size of the local domain; must be a `num_dims` large array. It won't be
  // owned.
  const int *size;
  // The number of MPI processes in each dimension. It is `num_dims` large. It
  // won't be owned.
  const int *num_mpi_procs;
  // The number of dimensions in the mesh.
  int num_dims;
  // The index of the dimension along which the equations are defined.
  int equation_dim;
  // The MPI coordinate of the current process along the dimension of the
  // equations.
  int mpi_coord;

  // Assumes that the number
  MpiSolverParams(MPI_Comm cartesian_communicator, int num_dims_,
                  const int *size_, int equation_dim_,
                  const int *num_mpi_procs_)
      : size{size_}, num_mpi_procs{num_mpi_procs_}, num_dims{num_dims_},
        equation_dim{equation_dim_} {
    int cart_rank;
    MPI_Comm_rank(cartesian_communicator, &cart_rank);
    std::vector<int> coords(num_dims_);
    MPI_Cart_coords(cartesian_communicator, cart_rank, num_dims_,
                    coords.data());
    std::vector<int> neighbours = {cart_rank};
    this->mpi_coord = coords[equation_dim];
    // Collect the processes in the same row/column
    for (int i = 1;
         i <= std::max(num_mpi_procs[equation_dim] - mpi_coord - 1, mpi_coord);
         ++i) {
      int prev, next;
      MPI_Cart_shift(cartesian_communicator, equation_dim, i, &prev, &next);
      if (i <= mpi_coord) {
        neighbours.push_back(prev);
      }
      if (i + mpi_coord < num_mpi_procs[equation_dim]) {
        neighbours.push_back(next);
      }
    }
    // This is needed, otherwise the communications hang
    std::sort(neighbours.begin(), neighbours.end());

    // Create new communicator for neighbours
    MPI_Group cart_group;
    MPI_Comm_group(cartesian_communicator, &cart_group);
    MPI_Group neighbours_group;
    MPI_Group_incl(cart_group, neighbours.size(), neighbours.data(),
                   &neighbours_group);
    MPI_Comm_create(cartesian_communicator, neighbours_group,
                    &this->communicator);
  }
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
  // The size of the equations / our domain
  const size_t local_eq_size = params.size[params.equation_dim];

  const MPI_Datatype real_datatype =
      std::is_same<REAL, double>::value ? MPI_DOUBLE : MPI_FLOAT;

  // Local modifications to the coefficients
  std::vector<REAL> aa(outer_size * eq_stride * local_eq_size),
      cc(outer_size * eq_stride * local_eq_size),
      dd(outer_size * eq_stride * local_eq_size);
  // MPI buffers (6 because 2 from each of the a, c and d coefficient arrays)
  const size_t comm_buf_size = 6 * outer_size * eq_stride;
  std::vector<REAL> send_buf(comm_buf_size),
      receive_buf(comm_buf_size * params.num_mpi_procs[params.equation_dim]);

  // Calculation
  // Forward pass
  for (size_t outer_ind = 0; outer_ind < outer_size; ++outer_ind) {
    // Start of the domain for the slice defined by `outer_ind` in the global
    // mesh along the dimension of the decomposition. (Note: the arrays only
    // contain the local data.)
    const size_t domain_start = outer_ind * local_eq_size * eq_stride;
    for (size_t local_eq_start = 0; local_eq_start < eq_stride;
         ++local_eq_start) {
      // The offset in the coefficient arrays a, b, c and d
      const size_t equation_offset = domain_start + local_eq_start;
      // The offset in the local arrays aa, cc and dd
      // Here the access is not strided, that's why it's different from
      // `equation_offset`.
      const size_t local_array_offset =
          (outer_ind * eq_stride + local_eq_start) * local_eq_size;

      thomas_forward<REAL>(
          a + equation_offset, b + equation_offset, c + equation_offset,
          d + equation_offset, nullptr, aa.data() + local_array_offset,
          cc.data() + local_array_offset, dd.data() + local_array_offset,
          local_eq_size, eq_stride);

      // The offset in the send and receive buffers
      const size_t comm_buf_offset =
          (outer_ind * eq_stride + local_eq_start) * 6;
      send_buf[comm_buf_offset + 0] = aa[local_array_offset + 0];
      send_buf[comm_buf_offset + 1] =
          aa[local_array_offset + local_eq_size - 1];
      send_buf[comm_buf_offset + 2] = cc[local_array_offset + 0];
      send_buf[comm_buf_offset + 3] =
          cc[local_array_offset + local_eq_size - 1];
      send_buf[comm_buf_offset + 4] = dd[local_array_offset + 0];
      send_buf[comm_buf_offset + 5] =
          dd[local_array_offset + local_eq_size - 1];
    }
  }

  // Communicate boundary results
  MPI_Allgather(send_buf.data(), comm_buf_size, real_datatype,
                receive_buf.data(), comm_buf_size, real_datatype,
                params.communicator);

  // Reduced system and backward pass
  for (size_t outer_ind = 0; outer_ind < outer_size; ++outer_ind) {
    // Start of the domain for the slice defined by `outer_ind` in the global
    // mesh along the dimension of the decomposition. (Note: the arrays only
    // contain the local data.)
    const size_t domain_start = outer_ind * local_eq_size * eq_stride;
    for (size_t local_eq_start = 0; local_eq_start < eq_stride;
         ++local_eq_start) {
      // The offset in the coefficient arrays a, b, c and d
      const size_t equation_offset = domain_start + local_eq_start;
      // The offset in the local arrays aa, cc and dd
      // Here the access is not strided, that's why it's different from
      // `equation_offset`.
      const size_t local_array_offset =
          (outer_ind * eq_stride + local_eq_start) * local_eq_size;

      // Reduced system
      std::vector<REAL> aa_r(2 * params.num_mpi_procs[params.equation_dim]),
          cc_r(2 * params.num_mpi_procs[params.equation_dim]),
          dd_r(2 * params.num_mpi_procs[params.equation_dim]);
      // The offset in the send and receive buffers
      const size_t comm_buf_offset =
          (outer_ind * eq_stride + local_eq_start) * 6;
      for (int i = 0; i < params.num_mpi_procs[params.equation_dim]; ++i) {
        aa_r[2 * i + 0] = receive_buf[comm_buf_size * i + comm_buf_offset + 0];
        aa_r[2 * i + 1] = receive_buf[comm_buf_size * i + comm_buf_offset + 1];
        cc_r[2 * i + 0] = receive_buf[comm_buf_size * i + comm_buf_offset + 2];
        cc_r[2 * i + 1] = receive_buf[comm_buf_size * i + comm_buf_offset + 3];
        dd_r[2 * i + 0] = receive_buf[comm_buf_size * i + comm_buf_offset + 4];
        dd_r[2 * i + 1] = receive_buf[comm_buf_size * i + comm_buf_offset + 5];
      }

      // indexing of cc_r, dd_r starts from 0
      // while indexing of aa_r starts from 1
      thomas_on_reduced<REAL>(aa_r.data(), cc_r.data(), dd_r.data(),
                              2 * params.num_mpi_procs[params.equation_dim], 1);

      dd[local_array_offset + 0] = dd_r[2 * params.mpi_coord];
      dd[local_array_offset + local_eq_size - 1] =
          dd_r[2 * params.mpi_coord + 1];
      thomas_backward<REAL>(aa.data() + local_array_offset,
                            cc.data() + local_array_offset,
                            dd.data() + local_array_offset, d + equation_offset,
                            local_eq_size, eq_stride);
    }
  }
}

#endif
