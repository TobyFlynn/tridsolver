#ifndef TRID_STRIDED_MULTIDIM_PCR_GPU_MPI__
#define TRID_STRIDED_MULTIDIM_PCR_GPU_MPI__

#include <cmath>

#include "trid_pcr_mpi_communication.hpp"

// Function adapted from trid_thomaspcr_large.hpp
template <typename REAL, int regStoreSize>
__device__ void loadDataIntoRegisters(REAL *regArray,  const REAL*  devArray, int tridiag, 
                                      int startElement, const int length, const int numTrids, 
                                      const int stride, const int batchSize, 
                                      const int batchStride, const REAL blank) {
  for(int i = 0; i < regStoreSize; i++) {
    int element = startElement + i;
    int memLoc = (tridiag % batchSize) + (element * stride) + (tridiag / batchSize) * batchStride;
    
    if(element < length && tridiag < numTrids) {
      regArray[i] = devArray[memLoc];
    } else {
      regArray[i] = blank;
    }
  }
}

template<typename REAL>
__device__ void storeDataFromRegisters(REAL *regArray,  REAL*  devArray, int tridiag, 
                                      int startElement, const int length, const int numTrids, 
                                      const int stride, const int batchSize, 
                                      const int batchStride, const int regStoreSize) {
  for(int i = 0; i < regStoreSize; i++) {
    int element = startElement + i;
    int memLoc = (tridiag % batchSize) + (element * stride) + (tridiag / batchSize) * batchStride;
    
    if(element < length && tridiag < numTrids) {
      devArray[memLoc] = regArray[i];
    }
  }
}

// Function that performs the modified Thomas forward pass on a GPU
// Adapted from code written by Jeremy Appleyard (see trid_thomaspcr_large.hpp)
template<typename REAL, int regStoreSize>
__global__ void batched_trid_forwards_kernel(const REAL* __restrict__ a, 
                                      const REAL* __restrict__ b, const REAL* __restrict__ c,
                                      const REAL* __restrict__ d, REAL* __restrict__ aa, 
                                      REAL* __restrict__ cc, REAL* __restrict__ dd, 
                                      REAL* __restrict__ aa_r, REAL* __restrict__ cc_r, 
                                      REAL* __restrict__ dd_r, const int length, 
                                      const int stride, const int numTrids, 
                                      const int batchSize, const int batchStride, 
                                      const int threadsPerTrid) {
  REAL a_reg[regStoreSize], b_reg[regStoreSize], c_reg[regStoreSize],
       d_reg[regStoreSize], aa_reg[regStoreSize], cc_reg[regStoreSize], dd_reg[regStoreSize]; 
  REAL bbi;
  
  int threadId_g = (blockIdx.x * blockDim.x) + threadIdx.x;
  int tridiag = threadId_g / threadsPerTrid;
  int threadId_l = (threadId_g - (tridiag * threadsPerTrid));
  int startElement = threadId_l * regStoreSize;
   
  loadDataIntoRegisters<REAL, regStoreSize>(&a_reg[0], a, tridiag, startElement, length, numTrids, stride, 
                                     batchSize, batchStride, (REAL)0.);
  
  loadDataIntoRegisters<REAL, regStoreSize>(b_reg, b, tridiag, startElement, length, numTrids, stride, 
                                     batchSize, batchStride, (REAL)1.);
  
  loadDataIntoRegisters<REAL, regStoreSize>(c_reg, c, tridiag, startElement, length, numTrids, stride, 
                                     batchSize, batchStride, (REAL)0.);
  
  loadDataIntoRegisters<REAL, regStoreSize>(d_reg, d, tridiag, startElement, length, numTrids, stride, 
                                     batchSize, batchStride, (REAL)0.);
  
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
    aa_r[tridiag + threadId_l * 2 * numTrids] = aa_reg[0];
    aa_r[tridiag + (threadId_l * 2 + 1) * numTrids] = aa_reg[n];
    cc_r[tridiag + threadId_l * 2 * numTrids] = aa_reg[0];
    cc_r[tridiag + (threadId_l * 2 + 1) * numTrids] = aa_reg[n];
    dd_r[tridiag + threadId_l * 2 * numTrids] = aa_reg[0];
    dd_r[tridiag + (threadId_l * 2 + 1) * numTrids] = aa_reg[n];
  }
  
  // Store aa, cc and dd values
  storeDataFromRegisters<REAL>(aa_reg, aa, tridiag, startElement, length, numTrids, 
                                      stride, batchSize, batchStride, regStoreSize);
  
  storeDataFromRegisters<REAL>(cc_reg, cc, tridiag, startElement, length, numTrids, 
                                      stride, batchSize, batchStride, regStoreSize);
  
  storeDataFromRegisters<REAL>(dd_reg, dd, tridiag, startElement, length, numTrids, 
                                      stride, batchSize, batchStride, regStoreSize);
}

// Will probably have to call each iteration separately as doubt you can make 
// MPI calls in CUDA
template <typename REAL>
void batched_trid_reduced(const REAL* __restrict__ aa_r, const REAL* __restrict__ cc_r, 
                          REAL* __restrict__ dd_r, const int numTrids, const int reducedSize, 
                          const int solvedim, const int threadsPerTrid, const int nBlocks, 
                          const int nThreads, const int size_g, trid_mpi_handle &mpi_handle) {
  // TODO see if a way of getting this without MPI reduce
  int reducedSize_g;
  if(solvedim == 1) {
    MPI_Allreduce(&reducedSize, &reducedSize_g, 1, MPI_INT, MPI_SUM, mpi_handle.y_comm);
  } else if(solvedim == 2) {
    MPI_Allreduce(&reducedSize, &reducedSize_g, 1, MPI_INT, MPI_SUM, mpi_handle.z_comm);
  }
  // Number of iterations for PCR
  int P = (int)ceil(log2(reducedSize_g));
  // Need arrays to store values from other MPI procs
  REAL *aa_r_s = NULL;
  REAL *cc_r_s = NULL;
  REAL *dd_r_s = NULL;
  cudaMalloc(&aa_r_s, sizeof(REAL) * reducedSize * numTrids * 2);
  cudaMalloc(&cc_r_s, sizeof(REAL) * reducedSize * numTrids * 2);
  cudaMalloc(&dd_r_s, sizeof(REAL) * reducedSize * numTrids * 2);
  
  // Needed for initial and final PCR stages as trid systems cannot be split across multiple blocks
  // in order to prevent race conditions
  int wholeTridThreads = (512 / threadsPerTrid) * threadsPerTrid;
  int wholeTridBlocks = (int)ceil((double)(threadsPerTrid * numTrids) / (double)wholeTridThreads);
  
  // Send and receive initial values
  getInitialValuesForPCR(aa_r, cc_r, dd_r, aa_r_s, cc_r_s, dd_r_s, solvedim, numTrids, 
                         threadsPerTrid, mpi_handle);
  
  // Perform initial step of PCR
  batched_trid_reduced_init_kernel<REAL><<<wholeTridBlocks, wholeTridThreads>>>(aa_r, cc_r, dd_r, 
                                  aa_r_s, cc_r_s, dd_r_s, numTrids, threadsPerTrid, reducedSize);
  
  for(int p = 1; p <= P; p++) {
    // s = 2^p
    int s = 1 << p;
    
    // Send and receive necessary values
    getValuesForPCR<REAL>(aa_r, cc_r, dd_r, aa_r_s, cc_r_s, dd_r_s, solvedim, numTrids, size_g, 
                          regStoreSize, mpi_handle);
    
    // Run PCR step on GPU
    batched_trid_reduced_kernel<REAL><<<nBlocks, nThreads>>>(aa_r, cc_r, dd_r, aa_r_s, cc_r_s, 
                                                  dd_r_s, numTrids, threadsPerTrid, reducedSize);
  }
  
  // Communicate boundary values for final step of PCR
  getFinalValuesForPCR<REAL>(dd_r, dd_r_s, solvedim, numTrids, mpi_handle);
  
  // Final part of PCR
  batched_trid_reduced_final_kernel<REAL><<<wholeTridBlocks, wholeTridThreads>>>(aa_r, cc_r, 
                                                      dd_r, dd_r_s, numTrids, threadsPerTrid);
  
  // Free memory
  cudaFree(aa_r_s);
  cudaFree(cc_r_s);
  cudaFree(dd_r_s);
}

template<typename REAL, int regStoreSize>
__global__ void batched_trid_backwards_kernel(const REAL* __restrict__ aa, 
                                  const REAL* __restrict__ cc, const REAL* __restrict__ dd, 
                                  const REAL* __restrict__ dd_r, REAL* __restrict__ d, 
                                  const int length, const int stride, const int numTrids, 
                                  const int batchSize, const int batchStride, 
                                  const int threadsPerTrid) {
  
  REAL aa_reg[regStoreSize], cc_reg[regStoreSize], dd_reg[regStoreSize];
  REAL dd_0, dd_n;
  
  int threadId_g = (blockIdx.x * blockDim.x) + threadIdx.x;
  int tridiag = threadId_g / threadsPerTrid;
  int threadId_l = (threadId_g - (tridiag * threadsPerTrid));
  int startElement = threadId_l * regStoreSize;
  
  loadDataIntoRegisters<REAL>(a_reg, aa, tridiag, startElement, length, numTrids, stride, 
                                     batchSize, batchStride, regStoreSize, (REAL)0.);
  
  loadDataIntoRegisters<REAL>(c_reg, cc, tridiag, startElement, length, numTrids, stride, 
                                     batchSize, batchStride, regStoreSize, (REAL)0.);
  
  loadDataIntoRegisters<REAL>(d_reg, dd, tridiag, startElement, length, numTrids, stride, 
                                     batchSize, batchStride, regStoreSize, (REAL)0.);
  
  int i = tridiag + numTrids * threadId_l * 2;
  dd_0 = dd_r[i];
  dd_n = dd_r[i + numTrids];
  
  dd_reg[0] = dd_0;
  dd_reg[regStoreSize - 1] = dd_n;
  
  for(int i = 1; i < regStoreSize - 1; i++) {
    dd_reg[i] = dd_reg[i] - aa_reg[i] * dd_0 - cc_reg[i] * dd_n;
  }
  
  storeDataFromRegisters<REAL>(dd_reg, d, tridiag, startElement, length, numTrids, 
                                      stride, batchSize, batchStride, regStoreSize);
}

template<typename REAL, int regStoreSize>
__global__ void batched_trid_backwardsInc_kernel(const REAL* __restrict__ aa, 
                                  const REAL* __restrict__ cc, const REAL* __restrict__ dd, 
                                  const REAL* __restrict__ dd_r, REAL* __restrict__ u, 
                                  const int length, const int stride, const int numTrids, 
                                  const int batchSize, const int batchStride, 
                                  const int threadsPerTrid) {
  
  REAL aa_reg[regStoreSize], cc_reg[regStoreSize], dd_reg[regStoreSize], u_reg[regStoreSize];
  REAL dd_0, dd_n;
  
  int threadId_g = (blockIdx.x * blockDim.x) + threadIdx.x;
  int tridiag = threadId_g / threadsPerTrid;
  int threadId_l = (threadId_g - (tridiag * threadsPerTrid));
  int startElement = threadId_l * regStoreSize;
  
  loadDataIntoRegisters<REAL>(a_reg, aa, tridiag, startElement, length, numTrids, stride, 
                                     batchSize, batchStride, regStoreSize, (REAL)0.);
  
  loadDataIntoRegisters<REAL>(c_reg, cc, tridiag, startElement, length, numTrids, stride, 
                                     batchSize, batchStride, regStoreSize, (REAL)0.);
  
  loadDataIntoRegisters<REAL>(d_reg, dd, tridiag, startElement, length, numTrids, stride, 
                                     batchSize, batchStride, regStoreSize, (REAL)0.);
  
  loadDataIntoRegisters<REAL>(u_reg, u, tridiag, startElement, length, numTrids, stride, 
                                     batchSize, batchStride, regStoreSize, (REAL)0.);
  
  int i = tridiag + numTrids * threadId_l * 2;
  dd_0 = dd_r[i];
  dd_n = dd_r[i + numTrids];
  
  u_reg[0] += dd_0;
  u_reg[regStoreSize - 1] += dd_n;
  
  for(int i = 1; i < regStoreSize - 1; i++) {
    u_reg[i] += dd_reg[i] - aa_reg[i] * dd_0 - cc_reg[i] * dd_n;
  }
  
  storeDataFromRegisters<REAL>(u_reg, u, tridiag, startElement, length, numTrids, 
                                      stride, batchSize, batchStride, regStoreSize);
}
#endif
