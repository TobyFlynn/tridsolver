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

#include "mpi.h"
#include "trid_mpi.h"
#include "cutil_inline.h"

template<typename REAL>
struct preproc_handle {
  REAL *halo_snd_x;
  REAL *halo_rcv_x;
  REAL *halo_snd_y;
  REAL *halo_rcv_y;
  REAL *halo_snd_z;
  REAL *halo_rcv_z;
  
  REAL *rcv_x;
  REAL *rcv_y;
  REAL *rcv_z;
  
  int rcv_size_x;
  int rcv_size_y;
  int rcv_size_z;
  
  REAL lambda;
};

struct preproc_kernel_arg_list {
  int pad_x;
  int start_g_x;
  int start_g_y;
  int start_g_z;
  int end_g_x;
  int end_g_y;
  int end_g_z;
  int nx_g;
  int ny_g;
  int nz_g;
  int nx;
  int ny;
  int nz;
};

template<typename REAL>
__global__ void preproc_mpi_cuda_kernel(REAL lambda, REAL *a, REAL *b, REAL *c, REAL *du, REAL *u, REAL *rcv_x, REAL *rcv_y, REAL *rcv_z, 
                                        const preproc_kernel_arg_list arg) {
  
  int i   = threadIdx.x + blockIdx.x*blockDim.x;
  int j   = threadIdx.y + blockIdx.y*blockDim.y;
  int ind = i + j*arg.pad_x; 

  // Is the thread in active region?
  int active = (i < arg.nx) && (j < arg.ny);
  
  REAL ux_1, ux_2, uy_1, uy_2, uz_1, uz_2;
  
  for(int k = 0; k < arg.nz; k++) {
    if(active) {
      if( (arg.start_g_x==0 && i==0) || 
          (arg.end_g_x==arg.nx_g-1 && i==arg.nx-1) ||
          (arg.start_g_y==0 && j==0) || 
          (arg.end_g_y==arg.ny_g-1 && j==arg.ny-1) ||
          (arg.start_g_z==0 && k==0) || 
          (arg.end_g_z==arg.nz_g-1 && k==arg.nz-1)) {

          du[ind] = 0.0f; // Dirichlet b.c.'s
          a[ind] = 0.0f;
          b[ind] = 1.0f;
          c[ind] = 0.0f;
        } else {
          if(i == 0) {
            ux_1 = rcv_x[1*arg.nz*arg.ny + k*arg.ny + j];
          } else {
            ux_1 = u[ind - 1];
          }
          
          if(i == arg.nx - 1) {
            ux_2 = rcv_x[0*arg.nz*arg.ny + k*arg.ny + j];
          } else {
            ux_2 = u[ind + 1];
          }
          
          if(j == 0) {
            uy_1 = rcv_y[1*arg.nz*arg.nx + k*arg.nx + i];
          } else {
            uy_1 = u[ind - arg.pad_x];
          }
          
          if(j == arg.ny - 1) {
            uy_2 = rcv_y[0*arg.nz*arg.nx + k*arg.nx + i];
          } else {
            uy_2 = u[ind + arg.pad_x];
          }
          
          if(k == 0) {
            uz_1 = rcv_z[1*arg.ny*arg.nx + j*arg.nx + i];
          } else {
            uz_1 = u[ind - arg.pad_x*arg.ny];
          }
          
          if(k == arg.nz - 1) {
            uz_2 = rcv_z[0*arg.ny*arg.nx + j*arg.nx + i];
          } else {
            uz_2 = u[ind + arg.pad_x*arg.ny];
          }
          
          du[ind] = lambda*( ux_1 + ux_2
                           + uy_1 + uy_2
                           + uz_1 + uz_2
                           - 6.0f * u[ind]);

          a[ind] = -0.5f * lambda;
          b[ind] =  1.0f + lambda;
          c[ind] = -0.5f * lambda;
        }
      ind += arg.pad_x*arg.ny;
    }
  }
}

template<typename REAL>
inline void preproc_mpi_cuda(preproc_handle<REAL> &pre_handle, trid_handle<REAL> &trid_handle, trid_mpi_handle &mpi_handle) {
  int   i, j, k, ind;
  double elapsed, timer = 0.0;
  //
  // calculate r.h.s. and set tri-diagonal coefficients
  //

  //timing_start(app.prof, &timer);
  
  int nx = trid_handle.size[0];
  int ny = trid_handle.size[1];
  int nz = trid_handle.size[2];
  
  int padx = trid_handle.pads[0];
  int pady = trid_handle.pads[1];
  int padz = trid_handle.pads[2];
  
  REAL *u = (REAL *) malloc(padx * pady * padz * sizeof(REAL));
  cudaSafeCall( cudaMemcpy(&u[0], &trid_handle.h_u[0], sizeof(FP) * padx * pady * padz, cudaMemcpyDeviceToHost) );
  
  // Gather halo
  // X boundary
  for(k = 0; k < nz; k++) {
    for(j = 0; j < ny; j++) {
      pre_handle.halo_snd_x[0*nz*ny + k*ny + j] = u[k*pady*padx + j*padx + 0];
      pre_handle.halo_snd_x[1*nz*ny + k*ny + j] = u[k*pady*padx + j*padx + nx - 1];
    }
  }
  
  // Y boundary
  for(k = 0; k < nz; k++) {
    for(i = 0; i < nx; i++) {
      pre_handle.halo_snd_y[0*nz*nx + k*nx + i] = u[k*pady*padx + i + 0];
      pre_handle.halo_snd_y[1*nz*nx + k*nx + i] = u[k*pady*padx + i + (padx)*(ny-1)];
    }
  }
  
  // Z boundary
  for(j = 0; j < ny; j++) {
    for(i = 0; i < nx; i++) {
      pre_handle.halo_snd_z[0*ny*nx + j*nx + i] = u[j*padx + i + 0];
      pre_handle.halo_snd_z[1*ny*nx + j*nx + i] = u[j*padx + i + padx*pady*(nz-1)];
    }
  }
  
  // Send and receive halo
  // Send X Left
  if(mpi_handle.coords[0] > 0) {
      // Convert destination coordinates of MPI node into the node's rank
      int dest_coords[3];
      dest_coords[0] = mpi_handle.coords[0] - 1;
      dest_coords[1] = mpi_handle.coords[1];
      dest_coords[2] = mpi_handle.coords[2];
      int destination_rank = 0;
      MPI_Cart_rank(mpi_handle.comm, dest_coords, &destination_rank);
      // Send the boundary data
      MPI_Send(&pre_handle.halo_snd_x[0*nz*ny], nz*ny, MPI_DOUBLE, destination_rank, 
               0, mpi_handle.comm);
  }
  // Receive
  if(mpi_handle.coords[0] < mpi_handle.pdims[0] - 1) {
      // Convert source coordinates of MPI node into the node's rank
      int source_coords[3];
      source_coords[0] = mpi_handle.coords[0] + 1;
      source_coords[1] = mpi_handle.coords[1];
      source_coords[2] = mpi_handle.coords[2];
      int source_rank = 0;
      MPI_Cart_rank(mpi_handle.comm, source_coords, &source_rank);
      MPI_Recv(&pre_handle.halo_rcv_x[0*nz*ny], nz*ny, MPI_DOUBLE, source_rank, 0,
               mpi_handle.comm, MPI_STATUS_IGNORE);
  }

  // Send X Right
  if(mpi_handle.coords[0] < mpi_handle.pdims[0] - 1) {
      // Convert destination coordinates of MPI node into the node's rank
      int dest_coords[3];
      dest_coords[0] = mpi_handle.coords[0] + 1;
      dest_coords[1] = mpi_handle.coords[1];
      dest_coords[2] = mpi_handle.coords[2];
      int destination_rank = 0;
      MPI_Cart_rank(mpi_handle.comm, dest_coords, &destination_rank);
      // Send the boundary data
      MPI_Send(&pre_handle.halo_snd_x[1*nz*ny], nz*ny, MPI_DOUBLE, destination_rank, 
               0, mpi_handle.comm);
  }
  // Receive
  if(mpi_handle.coords[0] > 0) {
      // Convert source coordinates of MPI node into the node's rank
      int source_coords[3];
      source_coords[0] = mpi_handle.coords[0] - 1;
      source_coords[1] = mpi_handle.coords[1];
      source_coords[2] = mpi_handle.coords[2];
      int source_rank = 0;
      MPI_Cart_rank(mpi_handle.comm, source_coords, &source_rank);
      MPI_Recv(&pre_handle.halo_rcv_x[1*nz*ny], nz*ny, MPI_DOUBLE, source_rank, 0,
               mpi_handle.comm, MPI_STATUS_IGNORE);
  }
  
  // Send Y Backwards
  if(mpi_handle.coords[1] > 0) {
      // Convert destination coordinates of MPI node into the node's rank
      int dest_coords[3];
      dest_coords[0] = mpi_handle.coords[0];
      dest_coords[1] = mpi_handle.coords[1] - 1;
      dest_coords[2] = mpi_handle.coords[2];
      int destination_rank = 0;
      MPI_Cart_rank(mpi_handle.comm, dest_coords, &destination_rank);
      // Send the boundary data
      MPI_Send(&pre_handle.halo_snd_y[0*nz*nx], nz*nx, MPI_DOUBLE, destination_rank, 
               0, mpi_handle.comm);
  }
  // Receive
  if(mpi_handle.coords[1] < mpi_handle.pdims[1] - 1) {
      // Convert source coordinates of MPI node into the node's rank
      int source_coords[3];
      source_coords[0] = mpi_handle.coords[0];
      source_coords[1] = mpi_handle.coords[1] + 1;
      source_coords[2] = mpi_handle.coords[2];
      int source_rank = 0;
      MPI_Cart_rank(mpi_handle.comm, source_coords, &source_rank);
      MPI_Recv(&pre_handle.halo_rcv_y[0*nz*nx], nz*nx, MPI_DOUBLE, source_rank, 0,
               mpi_handle.comm, MPI_STATUS_IGNORE);
  }
  
  // Send Y Forwards
  if(mpi_handle.coords[1] < mpi_handle.pdims[1] - 1) {
      // Convert destination coordinates of MPI node into the node's rank
      int dest_coords[3];
      dest_coords[0] = mpi_handle.coords[0];
      dest_coords[1] = mpi_handle.coords[1] + 1;
      dest_coords[2] = mpi_handle.coords[2];
      int destination_rank = 0;
      MPI_Cart_rank(mpi_handle.comm, dest_coords, &destination_rank);
      // Send the boundary data
      MPI_Send(&pre_handle.halo_snd_y[1*nz*nx], nz*nx, MPI_DOUBLE, destination_rank,
               0, mpi_handle.comm);
  }
  // Receive
  if(mpi_handle.coords[1] > 0) {
      // Convert source coordinates of MPI node into the node's rank
      int source_coords[3];
      source_coords[0] = mpi_handle.coords[0];
      source_coords[1] = mpi_handle.coords[1] - 1;
      source_coords[2] = mpi_handle.coords[2];
      int source_rank = 0;
      MPI_Cart_rank(mpi_handle.comm, source_coords, &source_rank);
      MPI_Recv(&pre_handle.halo_rcv_y[1*nz*nx], nz*nx, MPI_DOUBLE, source_rank, 0,
               mpi_handle.comm, MPI_STATUS_IGNORE);
  }
  
  // Send Z Below
  if(mpi_handle.coords[2] > 0) {
      // Convert destination coordinates of MPI node into the node's rank
      int dest_coords[3];
      dest_coords[0] = mpi_handle.coords[0];
      dest_coords[1] = mpi_handle.coords[1];
      dest_coords[2] = mpi_handle.coords[2] - 1;
      int destination_rank = 0;
      MPI_Cart_rank(mpi_handle.comm, dest_coords, &destination_rank);
      // Send the boundary data
      MPI_Send(&pre_handle.halo_snd_z[0*ny*nx], nx*ny, MPI_DOUBLE, destination_rank,
               0, mpi_handle.comm);
  }
  // Receive
  if(mpi_handle.coords[2] < mpi_handle.pdims[2] - 1) {
      // Convert source coordinates of MPI node into the node's rank
      int source_coords[3];
      source_coords[0] = mpi_handle.coords[0];
      source_coords[1] = mpi_handle.coords[1];
      source_coords[2] = mpi_handle.coords[2] + 1;
      int source_rank = 0;
      MPI_Cart_rank(mpi_handle.comm, source_coords, &source_rank);
      MPI_Recv(&pre_handle.halo_rcv_z[0*ny*nz], nx*ny, MPI_DOUBLE, source_rank, 0,
               mpi_handle.comm, MPI_STATUS_IGNORE);
  }
  
  // Send Z Above
  if(mpi_handle.coords[2] < mpi_handle.pdims[2] - 1) {
      // Convert destination coordinates of MPI node into the node's rank
      int dest_coords[3];
      dest_coords[0] = mpi_handle.coords[0];
      dest_coords[1] = mpi_handle.coords[1];
      dest_coords[2] = mpi_handle.coords[2] + 1;
      int destination_rank = 0;
      MPI_Cart_rank(mpi_handle.comm, dest_coords, &destination_rank);
      // Send the boundary data
      MPI_Send(&pre_handle.halo_snd_z[1*ny*nx], nx*ny, MPI_DOUBLE, destination_rank, 
               0, mpi_handle.comm);
  }
  // Receive
  if(mpi_handle.coords[2] > 0) {
      // Convert source coordinates of MPI node into the node's rank
      int source_coords[3];
      source_coords[0] = mpi_handle.coords[0];
      source_coords[1] = mpi_handle.coords[1];
      source_coords[2] = mpi_handle.coords[2] - 1;
      int source_rank = 0;
      MPI_Cart_rank(mpi_handle.comm, source_coords, &source_rank);
      MPI_Recv(&pre_handle.halo_rcv_z[1*ny*nx], nx*ny, MPI_DOUBLE, source_rank, 0, 
               mpi_handle.comm, MPI_STATUS_IGNORE);
  }
  
  free(u);
  
  //timing_end(app.prof, &timer, &app.elapsed_time[9], app.elapsed_name[9]);
  
  cudaSafeCall( cudaMemcpy(&pre_handle.rcv_x[0], &pre_handle.halo_rcv_x[0], sizeof(FP) * pre_handle.rcv_size_x, 
             cudaMemcpyHostToDevice) );
  cudaSafeCall( cudaMemcpy(&pre_handle.rcv_y[0], &pre_handle.halo_rcv_y[0], sizeof(FP) * pre_handle.rcv_size_y, 
             cudaMemcpyHostToDevice) );
  cudaSafeCall( cudaMemcpy(&pre_handle.rcv_z[0], &pre_handle.halo_rcv_z[0], sizeof(FP) * pre_handle.rcv_size_z, 
             cudaMemcpyHostToDevice) );
  
  preproc_kernel_arg_list arg;
  arg.pad_x = trid_handle.pads[0];
  arg.start_g_x = trid_handle.start_g[0];
  arg.start_g_y = trid_handle.start_g[1];
  arg.start_g_z = trid_handle.start_g[2];
  arg.end_g_x = trid_handle.end_g[0];
  arg.end_g_y = trid_handle.end_g[1];
  arg.end_g_z = trid_handle.end_g[2];
  arg.nx_g = trid_handle.size_g[0];
  arg.ny_g = trid_handle.size_g[1];
  arg.nz_g = trid_handle.size_g[2];
  arg.nx = trid_handle.size[0];
  arg.ny = trid_handle.size[1];
  arg.nz = trid_handle.size[2];
  
  dim3 dimGrid1(1+(nx-1)/32, 1+(ny-1)/4);
  dim3 dimBlock1(32,4);
  
  preproc_mpi_cuda_kernel<<<dimGrid1, dimBlock1>>>(pre_handle.lambda, trid_handle.a, 
                  trid_handle.b, trid_handle.c, trid_handle.du, trid_handle.h_u, 
                  pre_handle.rcv_x, pre_handle.rcv_y, pre_handle.rcv_z, arg);/*trid_handle.pads[0], 
                  trid_handle.start_g[0], trid_handle.start_g[1], trid_handle.start_g[2],
                  trid_handle.end_g[0], trid_handle.end_g[1], trid_handle.end_g[2],
                  trid_handle.size_g[0], trid_handle.size_g[1], trid_handle.size_g[2],
                  trid_handle.size[0], trid_handle.size[1], trid_handle.size[2]);*/
                  //trid_handle.start_g, trid_handle.end_g, trid_handle.size_g, trid_handle.size);
  cudaSafeCall( cudaPeekAtLastError() );
  cudaSafeCall( cudaDeviceSynchronize() );
  
  //timing_end(app.prof, &timer, &app.elapsed_time[10], app.elapsed_name[10]);
}
