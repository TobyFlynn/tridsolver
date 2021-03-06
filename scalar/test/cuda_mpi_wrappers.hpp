#ifndef TRID_CUDA_MPI_WRAPPERS_HPP
#define TRID_CUDA_MPI_WRAPPERS_HPP
#include "trid_mpi_cuda.hpp"

template <typename Float, bool INC = false>
tridStatus_t tridmtsvStridedBatchMPIWrapper(const MpiSolverParams &params,
                                            const Float *a, const Float *b,
                                            const Float *c, Float *d, Float *u,
                                            int ndim, int solvedim,
                                            const int *dims, const int *pads);

template <>
tridStatus_t tridmtsvStridedBatchMPIWrapper<float>(
    const MpiSolverParams &params, const float *a, const float *b,
    const float *c, float *d, float *u, int ndim, int solvedim, const int *dims,
    const int *pads) {
  return tridSmtsvStridedBatchMPI(params, a, b, c, d, u, ndim, solvedim, dims,
                                  pads);
}

template <>
tridStatus_t tridmtsvStridedBatchMPIWrapper<double>(
    const MpiSolverParams &params, const double *a, const double *b,
    const double *c, double *d, double *u, int ndim, int solvedim,
    const int *dims, const int *pads) {
  return tridDmtsvStridedBatchMPI(params, a, b, c, d, u, ndim, solvedim, dims,
                                  pads);
}

template <>
tridStatus_t tridmtsvStridedBatchMPIWrapper<float, true>(
    const MpiSolverParams &params, const float *a, const float *b,
    const float *c, float *d, float *u, int ndim, int solvedim, const int *dims,
    const int *pads) {
  return tridSmtsvStridedBatchIncMPI(params, a, b, c, d, u, ndim, solvedim,
                                     dims, pads);
}

template <>
tridStatus_t tridmtsvStridedBatchMPIWrapper<double, true>(
    const MpiSolverParams &params, const double *a, const double *b,
    const double *c, double *d, double *u, int ndim, int solvedim,
    const int *dims, const int *pads) {
  return tridDmtsvStridedBatchIncMPI(params, a, b, c, d, u, ndim, solvedim,
                                     dims, pads);
}

template <typename Float, bool INC = false>
tridStatus_t tridmtsvStridedBatchMPIWrapper(
    const MpiSolverParams &params, const Float *a, const int *a_pads,
    const Float *b, const int *b_pads, const Float *c, const int *c_pads,
    Float *d, const int *d_pads, Float *u, const int *u_pads, int ndim,
    int solvedim, const int *dims);

template <>
tridStatus_t tridmtsvStridedBatchMPIWrapper<float>(
    const MpiSolverParams &params, const float *a, const int *a_pads,
    const float *b, const int *b_pads, const float *c, const int *c_pads,
    float *d, const int *d_pads, float *u, const int *u_pads, int ndim,
    int solvedim, const int *dims) {
  return tridSmtsvStridedBatchPaddedMPI(params, a, a_pads, b, b_pads, c, c_pads,
                                        d, d_pads, u, u_pads, ndim, solvedim,
                                        dims);
}

template <>
tridStatus_t tridmtsvStridedBatchMPIWrapper<double>(
    const MpiSolverParams &params, const double *a, const int *a_pads,
    const double *b, const int *b_pads, const double *c, const int *c_pads,
    double *d, const int *d_pads, double *u, const int *u_pads, int ndim,
    int solvedim, const int *dims) {
  return tridDmtsvStridedBatchPaddedMPI(params, a, a_pads, b, b_pads, c, c_pads,
                                        d, d_pads, u, u_pads, ndim, solvedim,
                                        dims);
}

template <>
tridStatus_t tridmtsvStridedBatchMPIWrapper<float, true>(
    const MpiSolverParams &params, const float *a, const int *a_pads,
    const float *b, const int *b_pads, const float *c, const int *c_pads,
    float *d, const int *d_pads, float *u, const int *u_pads, int ndim,
    int solvedim, const int *dims) {
  return tridSmtsvStridedBatchPaddedIncMPI(params, a, a_pads, b, b_pads, c,
                                           c_pads, d, d_pads, u, u_pads, ndim,
                                           solvedim, dims);
}

template <>
tridStatus_t tridmtsvStridedBatchMPIWrapper<double, true>(
    const MpiSolverParams &params, const double *a, const int *a_pads,
    const double *b, const int *b_pads, const double *c, const int *c_pads,
    double *d, const int *d_pads, double *u, const int *u_pads, int ndim,
    int solvedim, const int *dims) {
  return tridDmtsvStridedBatchPaddedIncMPI(params, a, a_pads, b, b_pads, c,
                                           c_pads, d, d_pads, u, u_pads, ndim,
                                           solvedim, dims);
}

#endif /* ifndef TRID_CUDA_MPI_WRAPPERS_HPP */
