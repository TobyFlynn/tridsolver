#ifndef __ADI_MPI_CUDA_H
#define __ADI_MPI_CUDA_H

#include "mpi.h"
#include "trid_mpi_solver_params.hpp"

// ADI handle for MPI versions of the application
struct app_handle {
  FP *ax;
  FP *bx;
  FP *cx;
  FP *dx;
  FP *ux;

  FP *ay;
  FP *by;
  FP *cy;
  FP *dy;
  FP *uy;

  FP *az;
  FP *bz;
  FP *cz;
  FP *dz;
  FP *uz;

  int sizesX[3];
  int sizesY[3];
  int sizesZ[3];

  int *size_g;
  int *size;
  int *start_g;
  int *end_g;
  int *pads;

  int *pdims;
  int *coords;

  MPI_Comm comm;
  MpiSolverParams *params;
};

#endif
