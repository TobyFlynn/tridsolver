#include "trid_mpi.h"
#include "trid_strided_multidim_pcr_mpi.hpp"
#include "cutil_inline.h"

#include <cmath>

void setStartEnd(int *start, int *end, int coord, int numProcs, int numElements) {
  int tmp = numElements / numProcs;
  int remainder = numElements % numProcs;
  int total = 0;
  for(int i = 0; i < coord; i++) {
    if(i < remainder) {
      total += tmp + 1;
    } else {
      total += tmp;
    }
  }
  *start = total;
  if(coord < remainder) {
    *end = *start + tmp;
  } else {
    *end = *start + tmp -1;
  }
}

template<typename REAL>
void tridMultiDimBatchPCRInitMPI(trid_handle<REAL> &handle, trid_mpi_handle &mpi_handle, 
                                 int ndim, int *size) {
  // Get number of mpi procs and the rank of this mpi proc
  MPI_Comm_size(MPI_COMM_WORLD, &mpi_handle.procs);
  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_handle.rank);
  
  // Split into multi dim arrangement of mpi procs
  handle.ndim = ndim;
  mpi_handle.pdims    = (int *) calloc(handle.ndim, sizeof(int));
  mpi_handle.periodic = (int *) calloc(handle.ndim, sizeof(int)); //false
  mpi_handle.coords   = (int *) calloc(handle.ndim, sizeof(int));
  MPI_Dims_create(mpi_handle.procs, handle.ndim, mpi_handle.pdims);
  
  // Create cartecian mpi comm
  MPI_Cart_create(MPI_COMM_WORLD, handle.ndim, mpi_handle.pdims, mpi_handle.periodic, 0,  &mpi_handle.comm);
  
  // Get rand and coord of current mpi proc
  MPI_Comm_rank(mpi_handle.comm, &mpi_handle.my_cart_rank);
  MPI_Cart_coords(mpi_handle.comm, mpi_handle.my_cart_rank, handle.ndim, mpi_handle.coords);
  
  // TODO extend to other dimensions
  // Create separate comms for x, y and z dimensions
  int free_coords[3];
  free_coords[0] = 1;
  free_coords[1] = 0;
  free_coords[2] = 0;
  MPI_Cart_sub(mpi_handle.comm, free_coords, &mpi_handle.x_comm);
  MPI_Comm y_comm;
  free_coords[0] = 0;
  free_coords[1] = 1;
  free_coords[2] = 0;
  MPI_Cart_sub(mpi_handle.comm, free_coords, &mpi_handle.y_comm);
  MPI_Comm z_comm;
  free_coords[0] = 0;
  free_coords[1] = 0;
  free_coords[2] = 1;
  MPI_Cart_sub(mpi_handle.comm, free_coords, &mpi_handle.z_comm);
  
  // Store the global problem sizes
  handle.size_g = (int *) calloc(handle.ndim, sizeof(int));
  for(int i = 0; i < handle.ndim; i++) {
    handle.size_g[i] = size[i];
  }
  
  // Calculate size, padding, start and end for each dimension
  handle.size    = (int *) calloc(handle.ndim, sizeof(int));
  handle.pads    = (int *) calloc(handle.ndim, sizeof(int));
  handle.start_g = (int *) calloc(handle.ndim, sizeof(int));
  handle.end_g   = (int *) calloc(handle.ndim, sizeof(int));
  
  for(int i = 0; i < handle.ndim; i++) {
    setStartEnd(&handle.start_g[i], &handle.end_g[i], mpi_handle.coords[i], mpi_handle.pdims[i], 
                handle.size_g[i]);
    
    handle.size[i]    = handle.end_g[i] - handle.start_g[i] + 1;
    
    // Only pad the x dimension
    if(i == 0) {
      // TODO see what padding is needed for GPU
      //handle.pads[i] = (1 + ((handle.size[i] - 1) / SIMD_VEC)) * SIMD_VEC;
      handle.pads[i] = handle.size[i];
    } else {
      handle.pads[i] = handle.size[i];
    }
  }
  
  // Allocate memory for arrays
  int mem_size = sizeof(REAL);
  for(int i = 0; i < handle.ndim; i++) {
    mem_size *= handle.pads[i];
  }
  
  cudaSafeCall( cudaMalloc((void **)&handle.a, mem_size) );
  cudaSafeCall( cudaMalloc((void **)&handle.b, mem_size) );
  cudaSafeCall( cudaMalloc((void **)&handle.c, mem_size) );
  cudaSafeCall( cudaMalloc((void **)&handle.du, mem_size) );
  cudaSafeCall( cudaMalloc((void **)&handle.h_u, mem_size) );
  
  // Calculate reduced system sizes for each dimension
  handle.sys_len_l = (int *) calloc(handle.ndim, sizeof(int));
  handle.n_sys_g = (int *) calloc(handle.ndim, sizeof(int));
  handle.n_sys_l = (int *) calloc(handle.ndim, sizeof(int));
  
  for(int i = 0; i < handle.ndim; i++) {
    handle.sys_len_l[i] = mpi_handle.pdims[i] * 2;
    handle.n_sys_g[i] = 1;
    handle.n_sys_l[i] = 1;
    for(int j = 0; j < handle.ndim; j++) {
      if(j != i) {
        handle.n_sys_g[i] *= handle.size[j];
        handle.n_sys_l[i] *= handle.size[j];
      }
    }
  }
}

template<typename REAL>
void tridMultiDimBatchPCRCleanMPI(trid_handle<REAL> &handle, trid_mpi_handle &mpi_handle) {
  free(mpi_handle.pdims);
  free(mpi_handle.periodic);
  free(mpi_handle.coords);
  free(handle.size_g);
  free(handle.size);
  free(handle.start_g);
  free(handle.end_g);
  free(handle.sys_len_l);
  free(handle.n_sys_g);
  free(handle.n_sys_l);
  cudaFree(handle.a);
  cudaFree(handle.b);
  cudaFree(handle.c);
  cudaFree(handle.du);
  cudaFree(handle.h_u);
}

template<typename REAL, int INC>
void tridMultiDimBatchPCRSolveMPI(trid_handle<REAL> &handle, trid_mpi_handle &mpi_handle, 
                                  int solvedim) {
  // For now assume 1 MPI proc per GPU
  
  // Allocate aa, cc, dd
  REAL *aa = NULL;
  REAL *cc = NULL;
  REAL *dd = NULL;
  cudaMalloc((void **)&aa, sizeof(REAL) * handle.pads[0] * handle.size[1] * handle.size[2]);
  cudaMalloc((void **)&cc, sizeof(REAL) * handle.pads[0] * handle.size[1] * handle.size[2]);
  cudaMalloc((void **)&dd, sizeof(REAL) * handle.pads[0] * handle.size[1] * handle.size[2]);
  
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
    cudaMalloc((void **)&aa_r, sizeof(REAL) * reducedSize * numTrids);
    cudaMalloc((void **)&cc_r, sizeof(REAL) * reducedSize * numTrids);
    cudaMalloc((void **)&dd_r, sizeof(REAL) * reducedSize * numTrids);
    
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
    cudaMalloc((void **)&aa_r, sizeof(REAL) * reducedSize * numTrids);
    cudaMalloc((void **)&cc_r, sizeof(REAL) * reducedSize * numTrids);
    cudaMalloc((void **)&dd_r, sizeof(REAL) * reducedSize * numTrids);
    
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
    cudaMalloc((void **)&aa_r, sizeof(REAL) * reducedSize * numTrids);
    cudaMalloc((void **)&cc_r, sizeof(REAL) * reducedSize * numTrids);
    cudaMalloc((void **)&dd_r, sizeof(REAL) * reducedSize * numTrids);
    
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

template void tridMultiDimBatchPCRInitMPI<float>(trid_handle<float> &handle, 
                                              trid_mpi_handle &mpi_handle, int ndim, int *size);

template void tridMultiDimBatchPCRInitMPI<double>(trid_handle<double> &handle, 
                                              trid_mpi_handle &mpi_handle, int ndim, int *size);

template void tridMultiDimBatchPCRCleanMPI<float>(trid_handle<float> &handle, 
                                                  trid_mpi_handle &mpi_handle);

template void tridMultiDimBatchPCRCleanMPI<double>(trid_handle<double> &handle, 
                                                   trid_mpi_handle &mpi_handle);

template void tridMultiDimBatchPCRSolveMPI<float, 0>(trid_handle<float> &handle, 
                                                     trid_mpi_handle &mpi_handle, int solvedim);

template void tridMultiDimBatchPCRSolveMPI<double, 0>(trid_handle<double> &handle, 
                                                     trid_mpi_handle &mpi_handle, int solvedim);

template void tridMultiDimBatchPCRSolveMPI<float, 1>(trid_handle<float> &handle, 
                                                     trid_mpi_handle &mpi_handle, int solvedim);

template void tridMultiDimBatchPCRSolveMPI<double, 1>(trid_handle<double> &handle, 
                                                     trid_mpi_handle &mpi_handle, int solvedim);
