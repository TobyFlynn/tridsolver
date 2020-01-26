#ifndef TRID_STRIDED_MULTIDIM_PCR_GPU_MPI__
#define TRID_STRIDED_MULTIDIM_PCR_GPU_MPI__

#include <cmath>

// Function copied from trid_thomaspcr_large.hpp
template <typename REAL, int regStoreSize>
__device__ void loadDataIntoRegisters(REAL *regArray,  REAL*  devArray, int tridiag, 
                                      int subWarpIdx, const int length, const int numTrids, 
                                      const int stride, int subBatchID, int subBatchTrid, 
                                      const int subBatchSize, const REAL blank) {
  for (int i=0; i<regStoreSize; i++) {
    int element = subWarpIdx * regStoreSize + i;
    int gmemIdx = subBatchTrid + subBatchID * subBatchSize * stride + stride * element;

    if (element < length && tridiag < numTrids)
      regArray[i] = devArray[gmemIdx];
    else
      regArray[i] = blank;
  }
}

template<typename REAL, int regStoreSize>
__device__ void storeDataFromRegisters(REAL* regArray, REAL* devArray, int tridiag, 
                                       int subWarpIdx, const int length, const int numTrids,
                                       const int stride, int subBatchID, int subBatchTrid,
                                       const int subBatchSize) {
  for (int i=0; i<regStoreSize; i++) {   
    int element = subWarpIdx * regStoreSize + i;
    int gmemIdx = subBatchTrid + subBatchID * subBatchSize * stride + stride * element;
      
    if (element < length && tridiag < numTrids) {
      devArray[gmemIdx] = regArray[i];
    }
  }
}

// Function that performs the modified Thomas forward pass on a GPU
// Adapted from code written by Jeremy Appleyard (see trid_thomaspcr_large.hpp)
/*template<typename REAL, int regStoreSize, int blockSize, int blocksPerSMX, int tridSolveSize>
#if (__CUDA_ARCH__ >= 300)
__launch_bounds__(blockSize, blocksPerSMX)
#endif*/
template<typename REAL, int regStoreSize, int tridSolveSize>
__global__ void batched_trid_forwards_kernel(const REAL* __restrict__ a, 
                                      const REAL* __restrict__ b, const REAL* __restrict__ c,
                                      const REAL* __restrict__ d, REAL* __restrict__ aa, 
                                      REAL* __restrict__ cc, REAL* __restrict__ dd, 
                                      const int length, const int stride, const int numTrids, 
                                      const int subBatchSize, const int subBatchStride) {
  REAL a_reg[regStoreSize], b_reg[regStoreSize], c_reg[regStoreSize],
       d_reg[regStoreSize], aa_reg[regStoreSize], cc_reg[regStoreSize], dd_reg[regStoreSize]; 
  REAL bbi;
  
  int warpIdx = threadIdx.x / tridSolveSize;
  int subWarpIdx = threadIdx.x / (tridSolveSize / groupsPerTrid);
  int subThreadIdx = threadIdx.x % (tridSolveSize / groupsPerTrid);
   
  int tridiag = blockIdx.x * (tridSolveSize / groupsPerTrid) + subThreadIdx;
  
  int subBatchID = tridiag / subBatchSize;
  int subBatchTrid = tridiag % subBatchSize;
   
  loadDataIntoRegisters<REAL, regStoreSize>(a_reg, a, tridiag, subWarpIdx, length, 
                                            numTrids, stride, subBatchID, subBatchTrid, 
                                            subBatchSize, subBatchStride, (REAL)0.);
  
  loadDataIntoRegisters<REAL, regStoreSize>(b_reg, b, tridiag, subWarpIdx, length, 
                                            numTrids, stride, subBatchID, subBatchTrid, 
                                            subBatchSize, subBatchStride, (REAL)1.);
  
  loadDataIntoRegisters<REAL, regStoreSize>(c_reg, c, tridiag, subWarpIdx, length, 
                                            numTrids, stride, subBatchID, subBatchTrid, 
                                            subBatchSize, subBatchStride, (REAL)0.);
  
  loadDataIntoRegisters<REAL, regStoreSize>(d_reg, d, tridiag, subWarpIdx, length, 
                                            numTrids, stride, subBatchID, subBatchTrid, 
                                            subBatchSize, subBatchStride, (REAL)0.);
  
  // Reduce the system
  if (regStoreSize >= 2) {
    for (int i=0; i<2; i++) {
      bbi  = 1.0f / b_reg[i];
      dd_reg[i] = bbi * d_reg[i];
      aa_reg[i] = bbi * a_reg[i];
      cc_reg[i] = bbi * c_reg[i];
    }
     
    // The in-thread reduction here breaks down when the 
    // number of elements per thread drops below three. 
    if (regStoreSize >= 3) {
      for (int i=2; i<regStoreSize; i++) {
        bbi   = 1.0f / ( b_reg[i] - a_reg[i]*cc_reg[i-1] );
        dd_reg[i] =  bbi * ( d_reg[i] - a_reg[i]*dd_reg[i-1] );
        aa_reg[i] =  bbi * (          - a_reg[i]*aa_reg[i-1] );
        cc_reg[i] =  bbi *   c_reg[i];
      }

      for (int i=regStoreSize-3; i>0; i--) {
        dd_reg[i] =  dd_reg[i] - cc_reg[i]*dd_reg[i+1];
        aa_reg[i] =  aa_reg[i] - cc_reg[i]*aa_reg[i+1];
        cc_reg[i] =        - cc_reg[i]*cc_reg[i+1];
      }

      bbi = 1.0f / (1.0f - cc_reg[0]*aa_reg[1]);
      dd_reg[0] =  bbi * ( dd_reg[0] - cc_reg[0]*dd_reg[1] );
      aa_reg[0] =  bbi *   aa_reg[0];
      cc_reg[0] =  bbi * (       - cc_reg[0]*cc_reg[1] );
    }
  } else {
    bbi  = 1.0f / b_reg[0];
    dd_reg[0] = bbi * d_reg[0];
    aa_reg[0] = bbi * a_reg[0];
    cc_reg[0] = bbi * c_reg[0];      
  }
  
  // Store aa, cc and dd values
  storeDataFromRegisters<REAL, regStoreSize>(aa_reg, aa, tridiag, subWarpIdx, length, 
                                            numTrids, stride, subBatchID, subBatchTrid, 
                                            subBatchSize, subBatchStride);
  
  storeDataFromRegisters<REAL, regStoreSize>(cc_reg, cc, tridiag, subWarpIdx, length, 
                                            numTrids, stride, subBatchID, subBatchTrid, 
                                            subBatchSize, subBatchStride);
  
  storeDataFromRegisters<REAL, regStoreSize>(dd_reg, dd, tridiag, subWarpIdx, length, 
                                            numTrids, stride, subBatchID, subBatchTrid, 
                                            subBatchSize, subBatchStride);
}

template <typename REAL, int tridSolveSize, int blockSize>
void batched_trid_reduced(const REAL* __restrict__ aa, const REAL* __restrict__ cc, 
                          REAL* __restrict__ dd, trid_mpi_handle &mpi_handle) {
  
}

template <typename REAL, int tridSolveSize, int blockSize>
__global__ void batched_trid_reduced_kernel(REAL &am, REAL &cm, REAL &dm, REAL &ap, REAL &cp, 
                                            REAL &dp, volatile REAL *shared) {
  
}


template<typename REAL, int INC>
void tridMultiDimBatchPCRSolveMPI(trid_handle &handle, trid_mpi_handle &mpi_handle, int solvedim) {
  // For now assume 1 MPI proc per GPU
  
  // Allocate aa, cc, dd
  REAL *aa = NULL;
  REAL *cc = NULL;
  REAL *dd = NULL;
  cudaMalloc(&aa, sizeof(REAL) * handle.pads[0] * handle.size[1] * handle.size[2]);
  cudaMalloc(&cc, sizeof(REAL) * handle.pads[0] * handle.size[1] * handle.size[2]);
  cudaMalloc(&dd, sizeof(REAL) * handle.pads[0] * handle.size[1] * handle.size[2]);
  
  // TODO Work out grid and block sizes
  int nBlocks = 0;
  int nThreads = 0;
  
  if(solvedim == 0) {
    // For x dim might need to transpose (see single node version)
    // Transpose
    // Call tridMultiDimBatchPCRSolveMPI with new arguments
  } else if(solvedim == 1) {
    // Call forwards pass
    int numTrids = handle.size[0] * handle.size[2];
    int length = handle.size[1];
    int stride = handle.pads[0];
    int subBatchSize = handle.pads[0];
    int subBatchStride = handle.pads[0] * handle.size[1];
    // TODO understand most effective ways to increase these values as size increases
    int regStoreSize = 8;
    int tridSolveSize = 32;
    
    batched_trid_forwards_kernel<REAL, regStoreSize, tridSolveSize><<<nBlocks, nThreads>>>(
                                      handle.a, handle.b, handle.c, handle.du, aa, cc, dd, 
                                      length, stride, numTrids, subBatchSize, subBatchStride);
    
    // Call PCR reduced (modified to include MPI comm as reduced system will 
    // be spread over nodes)
    // Will probably have to call each iteration separately as doubt you can make 
    // MPI calls in CUDA
    int reducedSize = tridSolveSize * 2;
    // TODO see if a way of getting this without MPI reduce
    int reducedSize_g;
    MPI_Allreduce(&reducedSize, &reducedSize_g, 1, MPI_INT, MPI_SUM, mpi_handle.y_comm);
    // Number of iterations for PCR
    int p = (int)ceil(log2(reducedSize_g));
    
    // Call backwards pass
  } else if(solvedim == 2) {
    // Call forwards pass
    int numTrids = handle.size[0] * handle.size[1];
    int length = handle.size[2];
    int stride = handle.pads[0] * handle.size[1];
    int subBatchSize = handle.pads[0] * handle.size[1];
    int subBatchStride = 0;
    // TODO understand most effective ways to increase these values as size increases
    int regStoreSize = 8;
    int tridSolveSize = 32;
    
    batched_trid_forwards_kernel<REAL, regStoreSize, tridSolveSize><<<nBlocks, nThreads>>>(
                                       handle.a, handle.b, handle.c, handle.du, aa, cc, dd, 
                                       length, stride, numTrids, subBatchSize, subBatchStride);
    
    // Call PCR reduced (modified to include MPI comm as reduced system will 
    // be spread over nodes)
    // Will probably have to call each iteration separately as doubt you can make 
    // MPI calls in CUDA
    int reducedSize = tridSolveSize * 2;
    // TODO see if a way of getting this without MPI reduce
    int reducedSize_g;
    MPI_Allreduce(&reducedSize, &reducedSize_g, 1, MPI_INT, MPI_SUM, mpi_handle.z_comm);
    // Number of iterations for PCR
    int p = (int)ceil(log2(reducedSize_g));
    
    // Call backwards pass
  }
}

#endif
