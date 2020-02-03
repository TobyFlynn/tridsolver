#ifndef TRID_GPU_MPI_PCR__
#define TRID_GPU_MPI_PCR__

// Reduced system for each trid must not be split across multiple blocks in order to prevent race condition
template <typename REAL, int tridSolveSize, int blockSize>
__global__ void batched_trid_reduced_init_kernel(REAL* __restrict__ a, 
                                          REAL* __restrict__ c, REAL* __restrict__ d, 
                                          REAL* __restrict__  a_s, REAL* __restrict__ c_s, 
                                          REAL* __restrict__ d_s) {
  int threadId_g = (blockIdx.x * nThreads) + threadIdx.x;
  int tridiag = threadId_g / threadsPerTrid;
  int threadId_l = (threadId_g - (tridiag * threadsPerTrid));
  
  int i = tridiag + numTrids * threadId_l * 2;
  int i_p = i + numTrids;
  int i_m = i - numTrids;
  
  REAL a_p, a_m, c_p, c_m, d_p, d_m;
  
  if(threadId_l == 0) {
    int id_s = tridiag;
    a_m = a_s[id_s];
    c_m = c_s[id_s];
    d_m = d_s[id_s];
    
    a_p = a[i_p];
    c_p = c[i_p];
    d_p = d[i_p];
  } else if(threadId_l == threadsPerTrid - 1) {
    int id_s = reducedSize * numTrids + tridiag;
    a_p = a_s[id_s];
    c_p = c_s[id_s];
    d_p = d_s[id_s];
    
    a_m = a[i_m];
    c_m = c[i_m];
    d_m = d[i_m];
  } else {
    a_p = a[i_p];
    c_p = c[i_p];
    d_p = d[i_p];
    
    a_m = a[i_m];
    c_m = c[i_m];
    d_m = d[i_m];
  }
  
  __syncthreads();
  
  REAL r = (REAL)1.0 - a[i] * c_m - c[i] * a_p;
  r = (REAL)1.0 / r;
  d[i] = r * (d[i] - a[i] * d_m - c[i] * d_p);
  a[i] = -r * a[i] * a_m;
  c[i] = -r * c[i] * c_p;
}

template <typename REAL, int tridSolveSize, int blockSize>
__global__ void batched_trid_reduced_kernel(REAL* __restrict__ a, REAL* __restrict__ c, 
                                            REAL* __restrict__ d, REAL* __restrict__  a_s, 
                                            REAL* __restrict__ c_s, 
                                            REAL* __restrict__ d_s) {
  int threadId_g = (blockIdx.x * nThreads) + threadIdx.x;
  int tridiag = threadId_g / threadsPerTrid;
  int threadId_l = (threadId_g - (tridiag * threadsPerTrid));
  int i = tridiag + numTrids * threadId_l * 2;
  int minusId = tridiag + threadId_l * numTrids * 2;
  int plusId = reducedSize * numTrids + minusId;
  
  REAL r = (REAL)1.0 - a[i] * c_s[minusId] - c[i] * a_s[plusId];
  r = (REAL)1.0 / r;
  d[i] = r * (d[i] - a[i] * d_s[minusId] - c[i] * d_s[plusId]);
  a[i] = -r * a[i] * a_s[minusId];
  c[i] = -r * c[i] * c_s[plusId];
}

// Reduced system for each trid must not be split across multiple blocks in order to prevent race condition
template<typename REAL>
__global__ void batched_trid_reduced_final_kernel(REAL* __restrict__ a, REAL* __restrict__ c, 
                                                  REAL* __restrict__ d, REAL* __restrict__ d_s) {
  int threadId_g = (blockIdx.x * nThreads) + threadIdx.x;
  int tridiag = threadId_g / threadsPerTrid;
  int threadId_l = (threadId_g - (tridiag * threadsPerTrid));
  int i = tridiag + numTrids * threadId_l * 2;
  
  // To prevent race condition between threads
  REAL d_p2;
  if(threadId_l == threadsPerTrid - 1) {
    d_p2 = d_s[tridiag];
  } else {
    d_p2= d[i + 2 * numTrids];
  }
  
  __syncthreads();
  
  d[i] = d[i] - a[i] * d[i] - c[i] * d[i + numTrids]
  
  i +=  numTrids;
  
  d[i] = d[i] - a[i] * d[i] - c[i] * d_p2;
}

#endif
