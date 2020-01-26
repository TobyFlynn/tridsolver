#ifndef TRID_STRIDED_MULTIDIM_PCR_GPU_MPI__
#define TRID_STRIDED_MULTIDIM_PCR_GPU_MPI__

// Function copied from trid_thomaspcr_large.hpp
template <typename REAL, int regStoreSize, int tridSolveSize>
__device__ void loadDataIntoRegisters_large_strided(REAL *regArray,  REAL*  devArray, 
                                                    int tridiag, int subWarpIdx, 
                                                    int subThreadIdx, const int length, 
                                                    const int numTrids, const int stride,
                                                    int subBatchID, int subBatchTrid, 
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

// Function that performs the modified Thomas forward pass on a GPU
// Adapted from code written by Jeremy Appleyard (see trid_thomaspcr_large.hpp)
template<typename REAL, int regStoreSize, int blockSize, int blocksPerSMX, int tridSolveSize>
#if (__CUDA_ARCH__ >= 300)
__launch_bounds__(blockSize, blocksPerSMX)
#endif
__global__ void batched_trid_forwards(const REAL* __restrict__ a, const REAL* __restrict__ b,
                                      const REAL* __restrict__ c, const REAL* __restrict__ d,
                                      const int length, const int stride, const int numTrids
                                      const int subBatchSize, const int subBatchStride) {
  REAL a_reg[regStoreSize], b_reg[regStoreSize], c_reg[regStoreSize],
       d_reg[regStoreSize],aa[regStoreSize], cc[regStoreSize], dd[regStoreSize]; 
  REAL bbi;
  
  int warpIdx = threadIdx.x / tridSolveSize;
  int subWarpIdx = threadIdx.x / (tridSolveSize / groupsPerTrid);
  int subThreadIdx = threadIdx.x % (tridSolveSize / groupsPerTrid);
   
  int tridiag = blockIdx.x * (tridSolveSize / groupsPerTrid) + subThreadIdx;
  
  int subBatchID = tridiag / subBatchSize;
  int subBatchTrid = tridiag % subBatchSize;
   
  loadDataIntoRegisters_large_strided<REAL, regStoreSize, tridSolveSize>(a_reg, a, tridiag,
                                                      subWarpIdx, subThreadIdx, length, numTrids,
                                                      stride, subBatchID, subBatchTrid,
                                                      subBatchSize, subBatchStride, (REAL)0.);
  
  loadDataIntoRegisters_large_strided<REAL, regStoreSize, tridSolveSize>(b_reg, b, tridiag,
                                                      subWarpIdx, subThreadIdx, length, numTrids,
                                                      stride, subBatchID, subBatchTrid,
                                                      subBatchSize, subBatchStride, (REAL)1.);
  
  loadDataIntoRegisters_large_strided<REAL, regStoreSize, tridSolveSize>(c_reg, c, tridiag,
                                                      subWarpIdx, subThreadIdx, length, numTrids,
                                                      stride, subBatchID, subBatchTrid,
                                                      subBatchSize, subBatchStride, (REAL)0.);
  
  loadDataIntoRegisters_large_strided<REAL, regStoreSize, tridSolveSize>(d_reg, d, tridiag,
                                                      subWarpIdx, subThreadIdx, length, numTrids,
                                                      stride, subBatchID, subBatchTrid,
                                                      subBatchSize, subBatchStride, (REAL)0.);
  
  // Reduce the system
  if (regStoreSize >= 2) {
    for (int i=0; i<2; i++) {
      bbi  = 1.0f / b_reg[i];
      dd[i] = bbi * d_reg[i];
      aa[i] = bbi * a_reg[i];
      cc[i] = bbi * c_reg[i];
    }
     
    // The in-thread reduction here breaks down when the 
    // number of elements per thread drops below three. 
    if (regStoreSize >= 3) {
      for (int i=2; i<regStoreSize; i++) {
        bbi   = 1.0f / ( b_reg[i] - a_reg[i]*cc[i-1] );
        dd[i] =  bbi * ( d_reg[i] - a_reg[i]*dd[i-1] );
        aa[i] =  bbi * (          - a_reg[i]*aa[i-1] );
        cc[i] =  bbi *   c_reg[i];
      }

      for (int i=regStoreSize-3; i>0; i--) {
        dd[i] =  dd[i] - cc[i]*dd[i+1];
        aa[i] =  aa[i] - cc[i]*aa[i+1];
        cc[i] =        - cc[i]*cc[i+1];
      }

      bbi = 1.0f / (1.0f - cc[0]*aa[1]);
      dd[0] =  bbi * ( dd[0] - cc[0]*dd[1] );
      aa[0] =  bbi *   aa[0];
      cc[0] =  bbi * (       - cc[0]*cc[1] );
    }
  } else {
    bbi  = 1.0f / b_reg[0];
    dd[0] = bbi * d_reg[0];
    aa[0] = bbi * a_reg[0];
    cc[0] = bbi * c_reg[0];      
  }
  
  // TODO decide how to communicate aa, cc and dd values to pcr step
}

template<typename REAL, int INC>
void tridMultiDimBatchPCRSolveMPI(trid_handle &handle, trid_mpi_handle &mpi_handle, int solvedim) {
  // For now assume 1 MPI proc per GPU
  
  // Maybe allocate aa, cc, dd here as will be on GPU mem (so not in handle)
  
  // Work out grid and block sizes
  
  if(solvedim == 0) {
    // For x dim might need to transpose (see single node version)
    // Transpose
    // Call tridMultiDimBatchPCRSolveMPI with new arguments
  } else {
    // Call forwards pass
    
    // Call PCR reduced (modified to include MPI comm as reduced system will 
    // be spread over nodes)
    // Will probably have to call each iteration separately as doubt you can make 
    // MPI calls in CUDA
    
    // Call backwards pass
  }
}

#endif
