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

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <getopt.h>
#include <float.h>
#include <sys/time.h>

#include "preproc_mpi.hpp"
#include "trid_mpi_cpu.h"

#include "trid_common.h"

#include "omp.h"
//#include "offload.h"
#include "mpi.h"

#ifdef __MKL__
  //#include "lapacke.h"
  #include "mkl_lapacke.h"
  //#include "mkl.h"
#endif

#include "adi_mpi.h"

#define ROUND_DOWN(N,step) (((N)/(step))*step)
#define MIN(X,Y) ((X) < (Y) ? (X) : (Y))
#define MAX(X,Y) ((X) > (Y) ? (X) : (Y))

extern char *optarg;
extern int  optind, opterr, optopt;
static struct option options[] = {
  {"nx",   required_argument, 0,  0   },
  {"ny",   required_argument, 0,  0   },
  {"nz",   required_argument, 0,  0   },
  {"iter", required_argument, 0,  0   },
  {"opt",  required_argument, 0,  0   },
  {"prof", required_argument, 0,  0   },
  {"help", no_argument,       0,  'h' },
  {0,      0,                 0,  0   }
};

// Function for calculating local problem size for a MPI process, as well as its
// global start and end indices.
void setStartEnd(int *start, int *end, int coord, int numProcs, int numElements) {
  int tmp = numElements / numProcs;
  int remainder = numElements % numProcs;
  int total = 0;
  for(int i = 0; i < coord; i++) {
    if(i < remainder) {
      total += tmp + 1;
    } else {
      total += tmp;
    }
  }
  *start = total;
  if(coord < remainder) {
    *end = *start + tmp;
  } else {
    *end = *start + tmp - 1;
  }
}

/*
 * Print essential infromation on the use of the program
 */
void print_help() {
  printf("Please specify the ADI configuration, e.g.: \n$ ./adi_* -nx NX -ny NY -nz NZ -iter ITER [-opt CUDAOPT] -prof PROF\n");
  exit(0);
}

// Timing functions
inline double elapsed_time(double *et) {
  struct timeval t;
  double old_time = *et;

  gettimeofday( &t, (struct timezone *)0 );
  *et = t.tv_sec + t.tv_usec*1.0e-6;

  return *et - old_time;
}

inline void timing_start(double *timer) {
  elapsed_time(timer);
}

inline void timing_end(double *timer, double *elapsed_accumulate) {
  double elapsed = elapsed_time(timer);
  *elapsed_accumulate += elapsed;
}

// Function to add up a distributed array and print the result
void rms(char* name, FP* array, app_handle &handle) {
  //Sum the square of values in app.h_u
  double sum = 0.0;
  for(int k = 0; k < handle.size[2]; k++) {
    for(int j = 0; j < handle.size[1]; j++) {
      for(int i = 0; i < handle.size[0]; i++) {
        int ind = k * handle.pads[0] * handle.size[1] + j * handle.pads[0] + i;
        sum += array[ind];
      }
    }
  }

  double global_sum = 0.0;
  MPI_Allreduce(&sum, &global_sum,1, MPI_DOUBLE,MPI_SUM, handle.comm);

  if(handle.coords[0] == 0 && handle.coords[1] == 0 && handle.coords[2] == 0) {
    printf("%s sum = %.15g\n", name, global_sum);
  }

}

void decompose(int N, int M, int p, int *n, int *s) {
  int q = N / M;
  int r = N % M;
  *n = q + (r > p);
  *s = q * p + MIN(r, p);
}

int lsz(int N, MPI_Comm comm) {
  int size, rank, n, s;
  MPI_Comm_size(comm, &size);
  MPI_Comm_rank(comm, &rank);
  decompose(N, size, rank, &n, &s);
  return n;
}

// Initialize the ADI application
int init(app_handle &app, preproc_handle<FP> &pre_handle, int &iter, int argc, char* argv[]) {
  if( MPI_Init(&argc,&argv) != MPI_SUCCESS) { printf("MPI Couldn't initialize. Exiting"); exit(-1);}

  int nx_g = 256;
  int ny_g = 256;
  int nz_g = 256;
  iter = 10;
  int opt  = 0;
  int prof = 1;

  pre_handle.lambda = 1.0f;

  // Process arguments
  int opt_index = 0;
  while( getopt_long_only(argc, argv, "", options, &opt_index) != -1) {
    if(strcmp((char*)options[opt_index].name,"nx"  ) == 0) nx_g = atoi(optarg);
    if(strcmp((char*)options[opt_index].name,"ny"  ) == 0) ny_g = atoi(optarg);
    if(strcmp((char*)options[opt_index].name,"nz"  ) == 0) nz_g = atoi(optarg);
    if(strcmp((char*)options[opt_index].name,"iter") == 0) iter = atoi(optarg);
    if(strcmp((char*)options[opt_index].name,"opt" ) == 0) opt  = atoi(optarg);
    if(strcmp((char*)options[opt_index].name,"prof") == 0) prof = atoi(optarg);
    if(strcmp((char*)options[opt_index].name,"help") == 0) print_help();
  }

  // Allocate memory to store problem characteristics
  app.size_g  = (int *) calloc(3, sizeof(int));
  app.size    = (int *) calloc(3, sizeof(int));
  app.start_g = (int *) calloc(3, sizeof(int));
  app.end_g   = (int *) calloc(3, sizeof(int));
  app.pads    = (int *) calloc(3, sizeof(int));

  app.size_g[0] = nx_g;
  app.size_g[1] = ny_g;
  app.size_g[2] = nz_g;

  // Set up MPI for tridiagonal solver
  int procs, rank;
  MPI_Comm_size(MPI_COMM_WORLD, &procs);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  // Create 3D Cartesian MPI topology
  app.pdims     = (int *) calloc(2, sizeof(int));
  int *periodic = (int *) calloc(2, sizeof(int)); //false
  app.coords    = (int *) calloc(2, sizeof(int));
  MPI_Dims_create(procs, 2, app.pdims);

  // Create 3D Cartesian MPI communicator
  MPI_Cart_create(MPI_COMM_WORLD, 2, app.pdims, periodic, 0,  &app.comm);

  int my_cart_rank;

  MPI_Comm_rank(app.comm, &my_cart_rank);
  MPI_Cart_coords(app.comm, my_cart_rank, 2, app.coords);

  // Create MPI handle used by tridiagonal solver
  app.params = new MpiSolverParams(app.comm, 2, app.pdims);
  app.params->currentDim = 2;

  app.start_g[2] = 0;
  app.end_g[2] = app.size_g[2] - 1;
  app.size[2] = app.size_g[2];
  app.pads[2] = app.size[2];
  // Calculate local problem size for this MPI process
  for(int i = 0; i < 2; i++) {
    setStartEnd(&app.start_g[i], &app.end_g[i], app.coords[i], app.pdims[i], app.size_g[i]);
    app.size[i] = app.end_g[i] - app.start_g[i] + 1;
    /*if(i == 0) {
      app.pads[i] = (1 + ((app.size[i] - 1) / SIMD_VEC)) * SIMD_VEC;
    } else {*/
      app.pads[i] = app.size[i];
    //}
  }

  app.sizesZ[0] = lsz(app.size_g[0], app.params->communicators[0]);
  app.sizesZ[1] = lsz(app.size_g[1], app.params->communicators[1]);
  app.sizesZ[2] = app.size_g[2];
  app.sizesY[0] = lsz(app.size_g[0], app.params->communicators[0]);
  app.sizesY[1] = app.size_g[1];
  app.sizesY[2] = lsz(app.size_g[2], app.params->communicators[1]);
  app.sizesX[0] = app.size_g[0];
  app.sizesX[1] = lsz(app.size_g[1], app.params->communicators[0]);
  app.sizesX[2] = lsz(app.size_g[2], app.params->communicators[1]);

  free(periodic);

  if(rank==0) {
    printf("\nGlobal grid dimensions: %d x %d x %d\n",
           app.size_g[0], app.size_g[1], app.size_g[2]);

    printf("\nNumber of MPI procs in each dimenstion %d, %d\n",
           app.pdims[0], app.pdims[1]);
  }

  /*printf("Check parameters: SIMD_WIDTH = %d, sizeof(FP) = %d\n", SIMD_WIDTH, sizeof(FP));
  printf("Check parameters: nx_pad (padded) = %d\n", trid_handle.pads[0]);
  printf("Check parameters: nx = %d, x_start_g = %d, x_end_g = %d \n",
         trid_handle.size[0], trid_handle.start_g[0], trid_handle.end_g[0]);
  printf("Check parameters: ny = %d, y_start_g = %d, y_end_g = %d \n",
         trid_handle.size[1], trid_handle.start_g[1], trid_handle.end_g[1]);
  printf("Check parameters: nz = %d, z_start_g = %d, z_end_g = %d \n",
         trid_handle.size[2], trid_handle.start_g[2], trid_handle.end_g[2]);*/

  // Allocate memory for local section of problem
  int mem_sizeX = app.sizesX[0] * app.sizesX[1] * app.sizesX[2];
  int mem_sizeY = app.sizesY[0] * app.sizesY[1] * app.sizesY[2];
  int mem_sizeZ = app.sizesZ[0] * app.sizesZ[1] * app.sizesZ[2];

  app.ax = (FP *) _mm_malloc(mem_sizeX * sizeof(FP), SIMD_WIDTH);
  app.bx = (FP *) _mm_malloc(mem_sizeX * sizeof(FP), SIMD_WIDTH);
  app.cx = (FP *) _mm_malloc(mem_sizeX * sizeof(FP), SIMD_WIDTH);
  app.dx = (FP *) _mm_malloc(mem_sizeX * sizeof(FP), SIMD_WIDTH);
  app.ux = (FP *) _mm_malloc(mem_sizeX * sizeof(FP), SIMD_WIDTH);

  app.ay = (FP *) _mm_malloc(mem_sizeY * sizeof(FP), SIMD_WIDTH);
  app.by = (FP *) _mm_malloc(mem_sizeY * sizeof(FP), SIMD_WIDTH);
  app.cy = (FP *) _mm_malloc(mem_sizeY * sizeof(FP), SIMD_WIDTH);
  app.dy = (FP *) _mm_malloc(mem_sizeY * sizeof(FP), SIMD_WIDTH);
  app.uy = (FP *) _mm_malloc(mem_sizeY * sizeof(FP), SIMD_WIDTH);

  app.az = (FP *) _mm_malloc(mem_sizeZ * sizeof(FP), SIMD_WIDTH);
  app.bz = (FP *) _mm_malloc(mem_sizeZ * sizeof(FP), SIMD_WIDTH);
  app.cz = (FP *) _mm_malloc(mem_sizeZ * sizeof(FP), SIMD_WIDTH);
  app.dz = (FP *) _mm_malloc(mem_sizeZ * sizeof(FP), SIMD_WIDTH);
  app.uz = (FP *) _mm_malloc(mem_sizeZ * sizeof(FP), SIMD_WIDTH);

  // Initialize
  for(int k = 0; k < app.sizesZ[2]; k++) {
    for(int j = 0; j < app.sizesZ[1]; j++) {
      for(int i = 0; i < app.sizesZ[0]; i++) {
        int ind = k * app.sizesZ[0] * app.sizesZ[1] + j*app.sizesZ[0] + i;
        if( (app.start_g[0]==0 && i==0) ||
            (app.end_g[0]==app.size_g[0]-1 && i==app.size[0]-1) ||
            (app.start_g[1]==0 && j==0) ||
            (app.end_g[1]==app.size_g[1]-1 && j==app.size[1]-1) ||
            (app.start_g[2]==0 && k==0) ||
            (app.end_g[2]==app.size_g[2]-1 && k==app.size[2]-1)) {
          app.uz[ind] = 1.0f;
        } else {
          app.uz[ind] = 0.0f;
        }
      }
    }
  }

  // Allocate memory used in each iteration's preprocessing
  pre_handle.halo_snd_x = (FP*) _mm_malloc(2 * app.sizesZ[1] * app.sizesZ[2] * sizeof(FP), SIMD_WIDTH);
  pre_handle.halo_rcv_x = (FP*) _mm_malloc(2 * app.sizesZ[1] * app.sizesZ[2] * sizeof(FP), SIMD_WIDTH);
  pre_handle.halo_snd_y = (FP*) _mm_malloc(2 * app.sizesZ[0] * app.sizesZ[2] * sizeof(FP), SIMD_WIDTH);
  pre_handle.halo_rcv_y = (FP*) _mm_malloc(2 * app.sizesZ[0] * app.sizesZ[2] * sizeof(FP), SIMD_WIDTH);
  /*pre_handle.halo_snd_z = (FP*) _mm_malloc(2 * app.size[1] * app.size[0] * sizeof(FP), SIMD_WIDTH);
  pre_handle.halo_rcv_z = (FP*) _mm_malloc(2 * app.size[1] * app.size[0] * sizeof(FP), SIMD_WIDTH);*/

  return 0;

}

// Free memory used
void finalize(app_handle &app, preproc_handle<FP> &pre_handle) {
  _mm_free(pre_handle.halo_snd_x);
  _mm_free(pre_handle.halo_rcv_x);
  _mm_free(pre_handle.halo_snd_y);
  _mm_free(pre_handle.halo_rcv_y);
  /*_mm_free(pre_handle.halo_snd_z);
  _mm_free(pre_handle.halo_rcv_z);*/

  _mm_free(app.ax);
  _mm_free(app.bx);
  _mm_free(app.cx);
  _mm_free(app.dx);
  _mm_free(app.ux);

  _mm_free(app.ay);
  _mm_free(app.by);
  _mm_free(app.cy);
  _mm_free(app.dy);
  _mm_free(app.uy);

  _mm_free(app.az);
  _mm_free(app.bz);
  _mm_free(app.cz);
  _mm_free(app.dz);
  _mm_free(app.uz);

  free(app.size_g);
  free(app.size);
  free(app.start_g);
  free(app.end_g);
  free(app.pdims);
  free(app.coords);
  free(app.pads);

  delete app.params;
}

int main(int argc, char* argv[]) {
  app_handle app;
  preproc_handle<FP> pre_handle;
  int iter;
  // Initialize
  init(app, pre_handle, iter, argc, argv);

  // Declare and reset elapsed time counters
  double timer           = 0.0;
  double timer1          = 0.0;
  double timer2          = 0.0;
  double elapsed         = 0.0;
  double elapsed_total   = 0.0;
  double elapsed_preproc = 0.0;
  double elapsed_trid_x  = 0.0;
  double elapsed_trid_y  = 0.0;
  double elapsed_trid_z  = 0.0;

  timing_start(&timer1);

  // Iterate over specified number of time steps
  for(int it = 0; it < iter; it++) {
    // Preprocess
    timing_start(&timer);

    preproc_mpi<FP>(pre_handle, app);

    timing_end(&timer, &elapsed_preproc);

    //
    // perform tri-diagonal solves in x-direction
    //
    timing_start(&timer);
#if FPPREC == 0
    tridSmtsvStridedBatchMPI(*(app.params), app.ax, app.bx, app.cx, app.dx, app.ux,
                             app.ay, app.by, app.cy, app.dy, app.uy, app.az, app.bz,
                             app.cz, app.dz, app.uz, 3, 0, app.size, app.pads, app.size_g);
#else
    tridDmtsvStridedBatchMPI(*(app.params), app.ax, app.bx, app.cx, app.dx, app.ux,
                             app.ay, app.by, app.cy, app.dy, app.uy, app.az, app.bz,
                             app.cz, app.dz, app.uz, 3, 0, app.size, app.pads, app.size_g);
#endif
    timing_end(&timer, &elapsed_trid_x);

    //
    // perform tri-diagonal solves in y-direction
    //
    timing_start(&timer);
#if FPPREC == 0
    tridSmtsvStridedBatchMPI(*(app.params), app.ax, app.bx, app.cx, app.dx, app.ux,
                             app.ay, app.by, app.cy, app.dy, app.uy, app.az, app.bz,
                             app.cz, app.dz, app.uz, 3, 1, app.size, app.pads, app.size_g);
#else
    tridDmtsvStridedBatchMPI(*(app.params), app.ax, app.bx, app.cx, app.dx, app.ux,
                             app.ay, app.by, app.cy, app.dy, app.uy, app.az, app.bz,
                             app.cz, app.dz, app.uz, 3, 1, app.size, app.pads, app.size_g);
#endif
    timing_end(&timer, &elapsed_trid_y);

    //
    // perform tri-diagonal solves in z-direction
    //
    timing_start(&timer);
#if FPPREC == 0
    tridSmtsvStridedBatchIncMPI(*(app.params), app.ax, app.bx, app.cx, app.dx, app.ux,
                             app.ay, app.by, app.cy, app.dy, app.uy, app.az, app.bz,
                             app.cz, app.dz, app.uz, 3, 2, app.size, app.pads, app.size_g);
#else
    tridDmtsvStridedBatchIncMPI(*(app.params), app.ax, app.bx, app.cx, app.dx, app.ux,
                             app.ay, app.by, app.cy, app.dy, app.uy, app.az, app.bz,
                             app.cz, app.dz, app.uz, 3, 2, app.size, app.pads, app.size_g);
#endif
    timing_end(&timer, &elapsed_trid_z);
  }

  timing_end(&timer1, &elapsed_total);

  // Print sum of these arrays (basic error checking)
  rms("end h", app.uz, app);
  rms("end d", app.dz, app);

  MPI_Barrier(MPI_COMM_WORLD);

  // Print out timings of each section
  if(app.coords[0] == 0 && app.coords[1] == 0 && app.coords[2] == 0) {
    // Print execution times
    printf("Time per section: \n[total] \t[prepro] \t[trid_x] \t[trid_y] \t[trid_z]\n");
    printf("%e \t%e \t%e \t%e \t%e\n",
        elapsed_total,
        elapsed_preproc,
        elapsed_trid_x,
        elapsed_trid_y,
        elapsed_trid_z);
  }
  MPI_Barrier(MPI_COMM_WORLD);

  // Free memory
  finalize(app, pre_handle);

  MPI_Finalize();
  return 0;

}
