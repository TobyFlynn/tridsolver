#ifndef __TRID_CUDA_MPI_PCR_H
#define __TRID_CUDA_MPI_PCR_H

#include "trid_mpi.h"

template<typename REAL, int INC>
void tridMultiDimBatchPCRSolveMPI(trid_handle<REAL> &handle, trid_mpi_handle &mpi_handle, 
                                  int solvedim);

#endif