#ifndef TRID_STRIDED_MULTIDIM_PCR_GPU_MPI__
#define TRID_STRIDED_MULTIDIM_PCR_GPU_MPI__

#include <cmath>

// Function adapted from trid_thomaspcr_large.hpp
template <typename REAL, int regStoreSize>
__device__ void loadStridedDataIntoRegisters(REAL *regArray,  REAL*  devArray, int tridiag, 
                                      int startElement, const int length, const int numTrids, 
                                      const int stride, int batchSize, const int batchStride, 
                                      const REAL blank) {
  for(int i = 0; i < regStoreSize; i++) {
    int element = startElement + i;
    int memLoc = (tridiag % batchSize) + (element * stride) + (tridiag / batchSize) * batchStride;
    
    if(element < length && tridiag < numTrids) {
      regArray[i] = devArray[memLoc];
    } else {
      regArray[i] = blank;
    }
  }
  /*
  for (int i=0; i<regStoreSize; i++) {
    int element = subWarpIdx * regStoreSize + i;
    int gmemIdx = subBatchTrid + subBatchID * subBatchSize * stride + stride * element;

    if (element < length && tridiag < numTrids)
      regArray[i] = devArray[gmemIdx];
    else
      regArray[i] = blank;
  }*/
}

template<typename REAL, int regStoreSize>
__device__ void storeStridedDataFromRegisters(REAL *regArray,  REAL*  devArray, int tridiag, 
                                      int startElement, const int length, const int numTrids, 
                                      const int stride, int batchSize, const int batchStride) {
  for(int i = 0; i < regStoreSize; i++) {
    int element = startElement + i;
    int memLoc = (tridiag % batchSize) + (element * stride) + (tridiag / batchSize) * batchStride;
    
    if(element < length && tridiag < numTrids) {
      devArray[memLoc] = regArray[i];
    }
  }
  /*
  for (int i=0; i<regStoreSize; i++) {   
    int element = subWarpIdx * regStoreSize + i;
    int gmemIdx = subBatchTrid + subBatchID * subBatchSize * stride + stride * element;
      
    if (element < length && tridiag < numTrids) {
      devArray[gmemIdx] = regArray[i];
    }
  }*/
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
                                      REAL* __restrict__ aa_r, REAL* __restrict__ cc_r, 
                                      REAL* __restrict__ dd_r, const int length, 
                                      const int stride, const int numTrids, 
                                      const int batchSize, const int batchStride) {
  REAL a_reg[regStoreSize], b_reg[regStoreSize], c_reg[regStoreSize],
       d_reg[regStoreSize], aa_reg[regStoreSize], cc_reg[regStoreSize], dd_reg[regStoreSize]; 
  REAL bbi;
  
  int threadId_g = (blockIdx.x * nThreads) + threadIdx.x;
  int tridiag = threadId_g / threadsPerTrid;
  int startElement = (threadId_g - (tridiag * threadsPerTrid)) * regStoreSize;
  
  /*int warpIdx = threadIdx.x / tridSolveSize;
  int subWarpIdx = threadIdx.x / (tridSolveSize / groupsPerTrid);
  int subThreadIdx = threadIdx.x % (tridSolveSize / groupsPerTrid);
   
  int tridiag = blockIdx.x * (tridSolveSize / groupsPerTrid) + subThreadIdx;
  
  int subBatchID = tridiag / subBatchSize;
  int subBatchTrid = tridiag % subBatchSize;*/
   
  loadStridedDataIntoRegisters<REAL, regStoreSize>(a_reg, a, tridiag, startElement, length, 
                                                   numTrids, stride, batchSize, batchStride, 
                                                   (REAL)0.);
  
  loadStridedDataIntoRegisters<REAL, regStoreSize>(b_reg, b, tridiag, startElement, length, 
                                                   numTrids, stride, batchSize, batchStride, 
                                                   (REAL)0.);
  
  loadStridedDataIntoRegisters<REAL, regStoreSize>(c_reg, c, tridiag, startElement, length, 
                                                   numTrids, stride, batchSize, batchStride, 
                                                   (REAL)0.);
  
  loadStridedDataIntoRegisters<REAL, regStoreSize>(d_reg, d, tridiag, startElement, length, 
                                                   numTrids, stride, batchSize, batchStride, 
                                                   (REAL)0.);
  
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
  
  // Store reduced system
  if(tridiag < numTrids) {
    // TODO check if this is correct as values are padded?
    int n = regStoreSize - 1;
    /*if(startElement + n >= length) {
      n = startElement - lenght - 1;
    }*/
    aa_r[2 * tridiag]     = aa_reg[0];
    aa_r[2 * tridiag + 1] = aa_reg[n];
    cc_r[2 * tridiag]     = cc_reg[0];
    cc_r[2 * tridiag + 1] = cc_reg[n];
    dd_r[2 * tridiag]     = dd_reg[0];
    dd_r[2 * tridiag + 1] = dd_reg[n];
  }
  
  // Store aa, cc and dd values
  storeStridedDataFromRegisters<REAL, regStoreSize>(aa_reg, aa, tridiag, startElement, length, 
                                                    numTrids, stride, batchSize, batchStride);
  
  storeStridedDataFromRegisters<REAL, regStoreSize>(cc_reg, cc, tridiag, startElement, length, 
                                                    numTrids, stride, batchSize, batchStride);
  
  storeStridedDataFromRegisters<REAL, regStoreSize>(dd_reg, dd, tridiag, startElement, length, 
                                                    numTrids, stride, batchSize, batchStride);
}

template <typename REAL, int tridSolveSize, int blockSize>
void batched_trid_reduced(const REAL* __restrict__ aa_r, const REAL* __restrict__ cc_r, 
                          REAL* __restrict__ dd_r, const int s, trid_mpi_handle &mpi_handle) {
  
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
  
  if(solvedim == 0) {
    // For x dim might need to transpose (see single node version)
    // Transpose
    // Call tridMultiDimBatchPCRSolveMPI with new arguments
  } else if(solvedim == 1) {
    // Call forwards pass
    int numTrids = handle.size[0] * handle.size[2];
    int length = handle.size[1];
    int stride = handle.pads[0];
    int batchSize = handle.pads[0];
    int batchStride = handle.pads[0] * handle.size[1];
    int regStoreSize = 8;
    int threadsPerTrid = (int)ceil((double)handle.size[1] / (double)regStoreSize);
    
    // Work out number of blocks and threads needed
    int totalThreads = threadsPerTrid * numTrids;
    int nThreads = 512;
    int nBlocks = 1;
    if(totalThreads < 512) {
      nThreads = totalThreads;
    } else {
      nBlocks = (int)ceil((double)totalThreads / (double)nThreads);
    }
    
    int reducedSize = threadsPerTrid * numTrids * 2;
    // TODO change to one interwoven array once algorithm is working
    REAL *aa_r = NULL;
    REAL *cc_r = NULL;
    REAL *dd_r = NULL;
    cudaMalloc(&aa_r, sizeof(REAL) * reducedSize);
    cudaMalloc(&cc_r, sizeof(REAL) * reducedSize);
    cudaMalloc(&dd_r, sizeof(REAL) * reducedSize);
    
    batched_trid_forwards_kernel<REAL, regStoreSize, tridSolveSize><<<nBlocks, nThreads>>>(
                                      handle.a, handle.b, handle.c, handle.du, aa, cc, dd, aa_r, 
                                      cc_r, dd_r, length, stride, numTrids, batchSize, 
                                      batchStride);
    
    // Call PCR reduced (modified to include MPI comm as reduced system will 
    // be spread over nodes)
    // Will probably have to call each iteration separately as doubt you can make 
    // MPI calls in CUDA
    // TODO see if a way of getting this without MPI reduce
    int reducedSize_g;
    MPI_Allreduce(&reducedSize, &reducedSize_g, 1, MPI_INT, MPI_SUM, mpi_handle.y_comm);
    // Number of iterations for PCR
    int p = (int)ceil(log2(reducedSize_g));
    // Need arrays to store values from other MPI procs
    REAL *aa_r_minus = NULL;
    REAL *aa_r_plus  = NULL;
    REAL *cc_r_minus = NULL;
    REAL *cc_r_plus  = NULL;
    REAL *dd_r_minus = NULL;
    REAL *dd_r_plus  = NULL;
    cudaMalloc(&aa_r_minus, sizeof(REAL) * reducedSize);
    cudaMalloc(&aa_r_plus, sizeof(REAL) * reducedSize);
    cudaMalloc(&cc_r_minus, sizeof(REAL) * reducedSize);
    cudaMalloc(&cc_r_plus, sizeof(REAL) * reducedSize);
    cudaMalloc(&dd_r_minus, sizeof(REAL) * reducedSize);
    cudaMalloc(&dd_r_plus, sizeof(REAL) * reducedSize);
    
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
