#ifndef TRID_CUDA_MPI_WRAPPERS_HPP
#define TRID_CUDA_MPI_WRAPPERS_HPP
#include "trid_mpi_cuda.hpp"

template <typename Float>
tridStatus_t tridmtsvStridedBatchMPIWrapper(const MpiSolverParams &params,
                                            const Float *a, const Float *b,
                                            const Float *c, Float *d, Float *u,
                                            int *pads, int ndim, int solvedim,
                                            int *dims);

template <>
tridStatus_t tridmtsvStridedBatchMPIWrapper<float>(
    const MpiSolverParams &params, const float *a, const float *b,
    const float *c, float *d, float *u, int *pads, int ndim, int solvedim,
    int *dims) {
  return tridSmtsvStridedBatchMPI(params, a, b, c, d, u, pads, ndim, solvedim,
                                  dims);
}

template <>
tridStatus_t tridmtsvStridedBatchMPIWrapper<double>(
    const MpiSolverParams &params, const double *a, const double *b,
    const double *c, double *d, double *u, int *pads, int ndim, int solvedim,
    int *dims) {
  return tridDmtsvStridedBatchMPI(params, a, b, c, d, u, pads, ndim, solvedim,
                                  dims);
}

#endif /* ifndef TRID_CUDA_MPI_WRAPPERS_HPP */
