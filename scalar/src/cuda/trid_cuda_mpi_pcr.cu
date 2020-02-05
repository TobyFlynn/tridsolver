#include "trid_mpi.h"
#include "trid_strided_multidim_pcr_mpi.hpp"

#include <cmath>

template<typename REAL, int INC>
void tridMultiDimBatchPCRSolveMPI(trid_handle<REAL> &handle, trid_mpi_handle &mpi_handle, 
                                  int solvedim) {
  // For now assume 1 MPI proc per GPU
  
  // Allocate aa, cc, dd
  REAL *aa = NULL;
  REAL *cc = NULL;
  REAL *dd = NULL;
  cudaMalloc(&aa, sizeof(REAL) * handle.pads[0] * handle.size[1] * handle.size[2]);
  cudaMalloc(&cc, sizeof(REAL) * handle.pads[0] * handle.size[1] * handle.size[2]);
  cudaMalloc(&dd, sizeof(REAL) * handle.pads[0] * handle.size[1] * handle.size[2]);
  
  // TODO Copy memory from Host to GPU
  
  if(solvedim == 0) {
    // Call forwards pass
    const int numTrids = handle.size[1] * handle.size[2];
    const int length = handle.size[0];
    const int stride = 1;
    const int batchSize = 1;
    const int batchStride = handle.pads[0];
    const int regStoreSize = 8;
    const int threadsPerTrid = (int)ceil((double)length / (double)regStoreSize);
    
    // Work out number of blocks and threads needed
    int totalThreads = threadsPerTrid * numTrids;
    int nThreads = 512;
    int nBlocks = 1;
    if(totalThreads < 512) {
      nThreads = totalThreads;
    } else {
      nBlocks = (int)ceil((double)totalThreads / (double)nThreads);
    }
    
    int reducedSize = threadsPerTrid * 2;
    // TODO change to one interwoven array once algorithm is working
    REAL *aa_r = NULL;
    REAL *cc_r = NULL;
    REAL *dd_r = NULL;
    cudaMalloc(&aa_r, sizeof(REAL) * reducedSize * numTrids);
    cudaMalloc(&cc_r, sizeof(REAL) * reducedSize * numTrids);
    cudaMalloc(&dd_r, sizeof(REAL) * reducedSize * numTrids);
    
    // Call forwards pass
    batched_trid_forwards_kernel<REAL, regStoreSize><<<nBlocks, nThreads>>>(handle.a, handle.b, handle.c, 
                                handle.du, aa, cc, dd, aa_r, cc_r, dd_r, length, stride, 
                                numTrids, batchSize, batchStride, threadsPerTrid);
    
    // Call PCR reduced (modified to include MPI comm as reduced system will 
    // be spread over nodes)
    batched_trid_reduced<REAL, regStoreSize>(aa_r, cc_r, dd_r, numTrids, reducedSize, solvedim, threadsPerTrid, 
                               nBlocks, nThreads, length, mpi_handle);
    
    // Call backwards pass
    if(INC) {
      batched_trid_backwardsInc_kernel<REAL, regStoreSize><<<nBlocks, nThreads>>>(aa, cc, dd, dd_r, handle.h_u, 
                                                      length, stride, numTrids, batchSize, 
                                                      batchStride, threadsPerTrid);
    } else {
      batched_trid_backwards_kernel<REAL, regStoreSize><<<nBlocks, nThreads>>>(aa, cc, dd, dd_r, handle.du, 
                                                      length, stride, numTrids, batchSize, 
                                                      batchStride, threadsPerTrid);
    }
    
    // Free memory
    cudaFree(aa_r);
    cudaFree(cc_r);
    cudaFree(dd_r);
  } else if(solvedim == 1) {
    // Call forwards pass
    const int numTrids = handle.size[0] * handle.size[2];
    const int length = handle.size[1];
    const int stride = handle.pads[0];
    const int batchSize = handle.size[0];
    const int batchStride = handle.pads[0] * handle.size[1];
    const int regStoreSize = 8;
    const int threadsPerTrid = (int)ceil((double)length / (double)regStoreSize);
    
    // Work out number of blocks and threads needed
    int totalThreads = threadsPerTrid * numTrids;
    int nThreads = 512;
    int nBlocks = 1;
    if(totalThreads < 512) {
      nThreads = totalThreads;
    } else {
      nBlocks = (int)ceil((double)totalThreads / (double)nThreads);
    }
    
    int reducedSize = threadsPerTrid * 2;
    // TODO change to one interwoven array once algorithm is working
    REAL *aa_r = NULL;
    REAL *cc_r = NULL;
    REAL *dd_r = NULL;
    cudaMalloc(&aa_r, sizeof(REAL) * reducedSize * numTrids);
    cudaMalloc(&cc_r, sizeof(REAL) * reducedSize * numTrids);
    cudaMalloc(&dd_r, sizeof(REAL) * reducedSize * numTrids);
    
    // Call forwards pass
    batched_trid_forwards_kernel<REAL, regStoreSize><<<nBlocks, nThreads>>>(handle.a, handle.b, handle.c, 
                                handle.du, aa, cc, dd, aa_r, cc_r, dd_r, length, stride, 
                                numTrids, batchSize, batchStride, threadsPerTrid);
    
    // Call PCR reduced (modified to include MPI comm as reduced system will 
    // be spread over nodes)
    batched_trid_reduced<REAL, regStoreSize>(aa_r, cc_r, dd_r, numTrids, reducedSize, solvedim, threadsPerTrid, 
                               nBlocks, nThreads, length, mpi_handle);
    
    // Call backwards pass
    if(INC) {
      batched_trid_backwardsInc_kernel<REAL, regStoreSize><<<nBlocks, nThreads>>>(aa, cc, dd, dd_r, handle.h_u, 
                                                      length, stride, numTrids, batchSize, 
                                                      batchStride, threadsPerTrid);
    } else {
      batched_trid_backwards_kernel<REAL, regStoreSize><<<nBlocks, nThreads>>>(aa, cc, dd, dd_r, handle.du, 
                                                      length, stride, numTrids, batchSize, 
                                                      batchStride, threadsPerTrid);
    }
    
    // Free memory
    cudaFree(aa_r);
    cudaFree(cc_r);
    cudaFree(dd_r);
  } else if(solvedim == 2) {
    // Call forwards pass
    const int numTrids = handle.size[0] * handle.size[1];
    const int length = handle.size[2];
    const int stride = handle.pads[0] * handle.size[1];
    const int batchSize = handle.size[0];
    const int batchStride = handle.pads[0];
    const int regStoreSize = 8;
    const int threadsPerTrid = (int)ceil((double)length / (double)regStoreSize);
    
    // Work out number of blocks and threads needed
    int totalThreads = threadsPerTrid * numTrids;
    int nThreads = 512;
    int nBlocks = 1;
    if(totalThreads < 512) {
      nThreads = totalThreads;
    } else {
      nBlocks = (int)ceil((double)totalThreads / (double)nThreads);
    }
    
    int reducedSize = threadsPerTrid * 2;
    // TODO change to one interwoven array once algorithm is working
    REAL *aa_r = NULL;
    REAL *cc_r = NULL;
    REAL *dd_r = NULL;
    cudaMalloc(&aa_r, sizeof(REAL) * reducedSize * numTrids);
    cudaMalloc(&cc_r, sizeof(REAL) * reducedSize * numTrids);
    cudaMalloc(&dd_r, sizeof(REAL) * reducedSize * numTrids);
    
    // Call forwards pass
    batched_trid_forwards_kernel<REAL, regStoreSize><<<nBlocks, nThreads>>>(handle.a, handle.b, handle.c, 
                                handle.du, aa, cc, dd, aa_r, cc_r, dd_r, length, stride, 
                                numTrids, batchSize, batchStride, threadsPerTrid);
    
    // Call PCR reduced (modified to include MPI comm as reduced system will 
    // be spread over nodes)
    batched_trid_reduced<REAL, regStoreSize>(aa_r, cc_r, dd_r, numTrids, reducedSize, solvedim, threadsPerTrid, 
                               nBlocks, nThreads, length, mpi_handle);
    
    // Call backwards pass
    if(INC) {
      batched_trid_backwardsInc_kernel<REAL, regStoreSize><<<nBlocks, nThreads>>>(aa, cc, dd, dd_r, handle.h_u, 
                                                      length, stride, numTrids, batchSize, 
                                                      batchStride, threadsPerTrid);
    } else {
      batched_trid_backwards_kernel<REAL, regStoreSize><<<nBlocks, nThreads>>>(aa, cc, dd, dd_r, handle.du, 
                                                      length, stride, numTrids, batchSize, 
                                                      batchStride, threadsPerTrid);
    }
    
    // Free memory
    cudaFree(aa_r);
    cudaFree(cc_r);
    cudaFree(dd_r);
  }
}

template void tridMultiDimBatchPCRSolveMPI<float, 0>(trid_handle<float> &handle, 
                                                     trid_mpi_handle &mpi_handle, int solvedim);

template void tridMultiDimBatchPCRSolveMPI<double, 0>(trid_handle<double> &handle, 
                                                     trid_mpi_handle &mpi_handle, int solvedim);

template void tridMultiDimBatchPCRSolveMPI<float, 1>(trid_handle<float> &handle, 
                                                     trid_mpi_handle &mpi_handle, int solvedim);

template void tridMultiDimBatchPCRSolveMPI<double, 1>(trid_handle<double> &handle, 
                                                     trid_mpi_handle &mpi_handle, int solvedim);
