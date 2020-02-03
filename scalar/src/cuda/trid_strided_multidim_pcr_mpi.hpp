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

template<typename REAL>
void getFinalValuesForPCR(const REAL* __restrict__ d, REAL* __restrict__ d_s,
                          int solvedim, trid_mpi_handle &mpi_handle) {
  REAL *sndbuf = (REAL *) malloc(numTrids * sizeof(REAL));
  REAL *rcvbuf = (REAL *) malloc(numTrids * sizeof(REAL));
  
  cudaMemcpy(&sndbuf[0], &d[0], numTrids * sizeof(REAL), cudaMemcpyDeviceToHost);
  
  // Send
  if(solvedim == 1) {
    if(mpi_handle.coords[1] > 0) {
      // Convert destination coordinates of MPI node into the node's rank
      int dst_coords[3];
      dst_coords[0] = mpi_handle.coords[0];
      dst_coords[1] = mpi_handle.coords[1] - 1;
      dst_coords[2] = mpi_handle.coords[2];
      int dst_rank = 0;
      MPI_Cart_rank(mpi_handle.comm, dst_coords, &dst_rank);
      // Send the boundary data
      if(std::is_same<REAL, float>::value) {
        MPI_Send(&sndbuf[0], numTrids, MPI_FLOAT, dst_rank, 0, mpi_handle.comm);
      } else {
        MPI_Send(&sndbuf[0], numTrids, MPI_DOUBLE, dst_rank, 0, mpi_handle.comm);
      }
    }
  } else if(solvedim == 2) {
    if(mpi_handle.coords[2] > 0) {
      // Convert destination coordinates of MPI node into the node's rank
      int dst_coords[3];
      dst_coords[0] = mpi_handle.coords[0];
      dst_coords[1] = mpi_handle.coords[1];
      dst_coords[2] = mpi_handle.coords[2] - 1;
      int dst_rank = 0;
      MPI_Cart_rank(mpi_handle.comm, dst_coords, &dst_rank);
      // Send the boundary data
      if(std::is_same<REAL, float>::value) {
        MPI_Send(&sndbuf[0], numTrids, MPI_FLOAT, dst_rank, 0, mpi_handle.comm);
      } else {
        MPI_Send(&sndbuf[0], numTrids, MPI_DOUBLE, dst_rank, 0, mpi_handle.comm);
      }
    }
  }
  
  // Receive 
  if(solvedim == 1) {
    if(mpi_handle.coords[1] < mpi_handle.pdims[1] - 1) {
      // Convert src coordinates of MPI node into the node's rank
      int src_coords[3];
      src_coords[0] = mpi_handle.coords[0];
      src_coords[1] = mpi_handle.coords[1] + 1;
      src_coords[2] = mpi_handle.coords[2];
      int src_rank = 0;
      MPI_Cart_rank(mpi_handle.comm, src_coords, &src_rank);
      // Receive the boundary data
      if(std::is_same<REAL, float>::value) {
        MPI_Recv(&rcvbuf[0], numTrids, MPI_FLOAT, src_rank, 0, mpi_handle.comm, 
                 MPI_STATUS_IGNORE);
      } else {
        MPI_Recv(&rcvbuf[0], numTrids, MPI_DOUBLE, src_rank, 0, mpi_handle.comm, 
                 MPI_STATUS_IGNORE);
      }
    }
  } else if(solvedim == 2) {
    if(mpi_handle.coords[2] < mpi_handle.pdims[2] - 1) {
      // Convert destination coordinates of MPI node into the node's rank
      int src_coords[3];
      src_coords[0] = mpi_handle.coords[0];
      src_coords[1] = mpi_handle.coords[1];
      src_coords[2] = mpi_handle.coords[2] + 1;
      int src_rank = 0;
      MPI_Cart_rank(mpi_handle.comm, src_coords, &src_rank);
      // Receive the boundary data
      if(std::is_same<REAL, float>::value) {
        MPI_Recv(&rcvbuf[0], numTrids, MPI_FLOAT, src_rank, 0, mpi_handle.comm, 
                 MPI_STATUS_IGNORE);
      } else {
        MPI_Recv(&rcvbuf[0], numTrids, MPI_DOUBLE, src_rank, 0, mpi_handle.comm, 
                 MPI_STATUS_IGNORE);
      }
    }
  }
  
  // Copy to GPU
  cudaMemcpy(&d_s[0], &rcvbuf[0], numTrids, cudaMemcpyHostToDevice);
}

template<typename REAL>
void getInitialValuesForPCR(const REAL* __restrict__ a, const REAL* __restrict__ c, const REAL* __restrict__ d,
                            REAL* __restrict__ a_s, REAL* __restrict__ c_s, REAL* __restrict__ d_s,
                            int solvedim, trid_mpi_handle &mpi_handle) {
  // Buffer for a0, an, c0, cn, d0 and dn values for each trid system
  /*
   * sndbuf = | all 'a_0's | all 'c_0's | all 'd_0's | all 'a_n's | all 'c_n's | all 'd_n's |
   */
  REAL *sndbuf = (REAL *) malloc(3 * 2 * numTrids * sizeof(REAL));
  REAL *rcvbuf = (REAL *) malloc(3 * 2 * numTrids * sizeof(REAL));
  
  cudaMemcpy(&sndbuf[0], &a[0], numTrids * sizeof(REAL), cudaMemcpyDeviceToHost);
  cudaMemcpy(&sndbuf[3 * numTrids], &a[numTrids * (threadsPerTrid * 2 - 1)], 
             numTrids * sizeof(REAL), cudaMemcpyDeviceToHost);
  cudaMemcpy(&sndbuf[numTrids], &c[0], numTrids * sizeof(REAL), cudaMemcpyDeviceToHost);
  cudaMemcpy(&sndbuf[3 * numTrids + numTrids], &c[numTrids * (threadsPerTrid * 2 - 1)], 
             numTrids * sizeof(REAL), cudaMemcpyDeviceToHost);
  cudaMemcpy(&sndbuf[2 * numTrids], &d[0], numTrids * sizeof(REAL), cudaMemcpyDeviceToHost);
  cudaMemcpy(&sndbuf[3 * numTrids + 2 * numTrids], &d[numTrids * (threadsPerTrid * 2 - 1)], 
             numTrids * sizeof(REAL), cudaMemcpyDeviceToHost);
  
  // Send | all 'a_0's | all 'c_0's | all 'd_0's |
  if(solvedim == 1) {
    if(mpi_handle.coords[1] > 0) {
      // Convert destination coordinates of MPI node into the node's rank
      int dst_coords[3];
      dst_coords[0] = mpi_handle.coords[0];
      dst_coords[1] = mpi_handle.coords[1] - 1;
      dst_coords[2] = mpi_handle.coords[2];
      int dst_rank = 0;
      MPI_Cart_rank(mpi_handle.comm, dst_coords, &dst_rank);
      // Send the boundary data
      if(std::is_same<REAL, float>::value) {
        MPI_Send(&sndbuf[0], numTrids * 3, MPI_FLOAT, dst_rank, 0, mpi_handle.comm);
      } else {
        MPI_Send(&sndbuf[0], numTrids * 3, MPI_DOUBLE, dst_rank, 0, mpi_handle.comm);
      }
    }
  } else if(solvedim == 2) {
    if(mpi_handle.coords[2] > 0) {
      // Convert destination coordinates of MPI node into the node's rank
      int dst_coords[3];
      dst_coords[0] = mpi_handle.coords[0];
      dst_coords[1] = mpi_handle.coords[1];
      dst_coords[2] = mpi_handle.coords[2] - 1;
      int dst_rank = 0;
      MPI_Cart_rank(mpi_handle.comm, dst_coords, &dst_rank);
      // Send the boundary data
      if(std::is_same<REAL, float>::value) {
        MPI_Send(&sndbuf[0], numTrids * 3, MPI_FLOAT, dst_rank, 0, mpi_handle.comm);
      } else {
        MPI_Send(&sndbuf[0], numTrids * 3, MPI_DOUBLE, dst_rank, 0, mpi_handle.comm);
      }
    }
  }
  
  // Receive 
  if(solvedim == 1) {
    if(mpi_handle.coords[1] < mpi_handle.pdims[1] - 1) {
      // Convert src coordinates of MPI node into the node's rank
      int src_coords[3];
      src_coords[0] = mpi_handle.coords[0];
      src_coords[1] = mpi_handle.coords[1] + 1;
      src_coords[2] = mpi_handle.coords[2];
      int src_rank = 0;
      MPI_Cart_rank(mpi_handle.comm, src_coords, &src_rank);
      // Receive the boundary data
      if(std::is_same<REAL, float>::value) {
        MPI_Recv(&rcvbuf[0], numTrids * 3, MPI_FLOAT, src_rank, 0, mpi_handle.comm, 
                 MPI_STATUS_IGNORE);
      } else {
        MPI_Recv(&rcvbuf[0], numTrids * 3, MPI_DOUBLE, src_rank, 0, mpi_handle.comm, 
                 MPI_STATUS_IGNORE);
      }
    }
  } else if(solvedim == 2) {
    if(mpi_handle.coords[2] < mpi_handle.pdims[2] - 1) {
      // Convert destination coordinates of MPI node into the node's rank
      int src_coords[3];
      src_coords[0] = mpi_handle.coords[0];
      src_coords[1] = mpi_handle.coords[1];
      src_coords[2] = mpi_handle.coords[2] + 1;
      int src_rank = 0;
      MPI_Cart_rank(mpi_handle.comm, src_coords, &src_rank);
      // Receive the boundary data
      if(std::is_same<REAL, float>::value) {
        MPI_Recv(&rcvbuf[0], numTrids * 3, MPI_FLOAT, src_rank, 0, mpi_handle.comm, 
                 MPI_STATUS_IGNORE);
      } else {
        MPI_Recv(&rcvbuf[0], numTrids * 3, MPI_DOUBLE, src_rank, 0, mpi_handle.comm, 
                 MPI_STATUS_IGNORE);
      }
    }
  }
  
  // Send | all 'a_n's | all 'c_n's | all 'd_n's |
  if(solvedim == 1) {
    if(mpi_handle.coords[1] < mpi_handle.pdims[1] - 1) {
      // Convert destination coordinates of MPI node into the node's rank
      int dst_coords[3];
      dst_coords[0] = mpi_handle.coords[0];
      dst_coords[1] = mpi_handle.coords[1] + 1;
      dst_coords[2] = mpi_handle.coords[2];
      int dst_rank = 0;
      MPI_Cart_rank(mpi_handle.comm, dst_coords, &dst_rank);
      // Send the boundary data
      if(std::is_same<REAL, float>::value) {
        MPI_Send(&sndbuf[3 * numTrids], numTrids * 3, MPI_FLOAT, dst_rank, 0, mpi_handle.comm);
      } else {
        MPI_Send(&sndbuf[3 * numTrids], numTrids * 3, MPI_DOUBLE, dst_rank, 0, mpi_handle.comm);
      }
    }
  } else if(solvedim == 2) {
    if(mpi_handle.coords[2] < mpi_handle.pdims[2] - 1) {
      // Convert destination coordinates of MPI node into the node's rank
      int dst_coords[3];
      dst_coords[0] = mpi_handle.coords[0];
      dst_coords[1] = mpi_handle.coords[1];
      dst_coords[2] = mpi_handle.coords[2] + 1;
      int dst_rank = 0;
      MPI_Cart_rank(mpi_handle.comm, dst_coords, &dst_rank);
      // Send the boundary data
      if(std::is_same<REAL, float>::value) {
        MPI_Send(&sndbuf[3 * numTrids], numTrids * 3, MPI_FLOAT, dst_rank, 0, mpi_handle.comm);
      } else {
        MPI_Send(&sndbuf[3 * numTrids], numTrids * 3, MPI_DOUBLE, dst_rank, 0, mpi_handle.comm);
      }
    }
  }
  
  // Receive 
  if(solvedim == 1) {
    if(mpi_handle.coords[1] > 0) {
      // Convert src coordinates of MPI node into the node's rank
      int src_coords[3];
      src_coords[0] = mpi_handle.coords[0];
      src_coords[1] = mpi_handle.coords[1] - 1;
      src_coords[2] = mpi_handle.coords[2];
      int src_rank = 0;
      MPI_Cart_rank(mpi_handle.comm, src_coords, &src_rank);
      // Receive the boundary data
      if(std::is_same<REAL, float>::value) {
        MPI_Recv(&rcvbuf[0], numTrids * 3, MPI_FLOAT, src_rank, 0, mpi_handle.comm, 
                 MPI_STATUS_IGNORE);
      } else {
        MPI_Recv(&rcvbuf[0], numTrids * 3, MPI_DOUBLE, src_rank, 0, mpi_handle.comm, 
                 MPI_STATUS_IGNORE);
      }
    }
  } else if(solvedim == 2) {
    if(mpi_handle.coords[2] > 0) {
      // Convert destination coordinates of MPI node into the node's rank
      int src_coords[3];
      src_coords[0] = mpi_handle.coords[0];
      src_coords[1] = mpi_handle.coords[1];
      src_coords[2] = mpi_handle.coords[2] - 1;
      int src_rank = 0;
      MPI_Cart_rank(mpi_handle.comm, src_coords, &src_rank);
      // Receive the boundary data
      if(std::is_same<REAL, float>::value) {
        MPI_Recv(&rcvbuf[3 * numTrids], numTrids * 3, MPI_FLOAT, src_rank, 0, mpi_handle.comm, 
                 MPI_STATUS_IGNORE);
      } else {
        MPI_Recv(&rcvbuf[3 * numTrids], numTrids * 3, MPI_DOUBLE, src_rank, 0, mpi_handle.comm, 
                 MPI_STATUS_IGNORE);
      }
    }
  }

  // Transfer back to GPU
  cudaMemcpy(&a_s[0], &rcvbuf[0], numTrids, cudaMemcpyHostToDevice);
  cudaMemcpy(&a_s[reducedSize * numTrids], &rcvbuf[3 * numTrids], numTrids, 
             cudaMemcpyHostToDevice);
  
  cudaMemcpy(&c_s[0], &rcvbuf[numTrids], numTrids, cudaMemcpyHostToDevice);
  cudaMemcpy(&c_s[reducedSize * numTrids], &rcvbuf[3 * numTrids + numTrids], numTrids, 
             cudaMemcpyHostToDevice);
  
  cudaMemcpy(&d_s[0], &rcvbuf[2 * numTrids], numTrids, cudaMemcpyHostToDevice);
  cudaMemcpy(&d_s[reducedSize * numTrids], &rcvbuf[3 * numTrids + 2 * numTrids], numTrids, 
             cudaMemcpyHostToDevice);
}

template<typename REAL>
void getValuesForPCR(const REAL* __restrict__ a, const REAL* __restrict__ c, const REAL* __restrict__ d,
                     REAL* __restrict__ a_s, REAL* __restrict__ c_s, REAL* __restrict__ d_s,
                     int solvedim, trid_mpi_handle &mpi_handle) {
  // Get sizes for each proc and the size of the reduced system
  int tmp = numElements / numProcs;
  int remainder = numElements % numProcs;
  int sizes[mpi_handle.pdims[solvedim]];
  int reducedSizes[mpi_handle.pdims[solvedim]];
  for(int i = 0; i < mpi_handle.pdims[solvedim]; i++) {
    if(i < remainder) {
      sizes[i] = tmp + 1;
    } else {
      sizes[i] = tmp;
    }
    reducedSizes[i] = 2 * (int)ceil((double)sizes[i] / (double)regStoreSize);
  }
  
  // Get global start indices for each proc
  int reducedStart_g[mpi_handle.pdims[solvedim]];
  int total = 0;
  for(int i = 0; i < mpi_handle.pdims[solvedim]; i++) {
    reducedStart_g[i] = total;
    total += reducedSizes[i];
  }
  
  int this_proc_start_g = reducedStart_g[mpi_handle.coords[solvedim]];
  int this_proc_end_g = this_proc_start_g + numTrids * reducedSizes[mpi_handle.coords[solvedim]];
  int this_proc_reduced_size = reducedSizes[mpi_handle.coords[solvedim]];
  
  // Allocate send buffers (will only ever send to max 2 MPI procs)
  REAL *sndbuf_1 = (REAL *) malloc(3 * this_proc_reduced_size * numTrids * sizeof(REAL));
  REAL *sndbuf_2 = (REAL *) malloc(3 * this_proc_reduced_size * numTrids * sizeof(REAL));
  bool usedFirstSndBuf = false;
  
  /*
   * =====================================
   * -S ELEMENTS
   * =====================================
   */
  
  // Only need to check procs that are 'above' the current MPI proc for "-s" elements
  for(int i = mpi_handle.coords[solvedim] + 1; i < mpi_handle.pdims[solvedim]; i++) {
    int send_start_g = reducedStart_g[i] - s;
    int send_end_g = send_start_g + reducedSizes[i];
     
    // Check if need to send data to this MPI proc
    if((send_start_g <= this_proc_end_g && send_start_g >= this_proc_start_g)
        || (send_end_g <= this_proc_end_g && send_end_g >= this_proc_start_g)
        || (send_start_g < this_proc_start_g && send_end_g > this_proc_end_g)) {
      // Get rank of MPI proc to send to
      // Convert destination coordinates of MPI node into the node's rank
      int dst_coords[3];
      dst_coords[0] = mpi_handle.coords[0];
      dst_coords[1] = mpi_handle.coords[1];
      dst_coords[2] = mpi_handle.coords[2];
      dst_coords[solvedim] = i;
      int dst_rank = 0;
      MPI_Cart_rank(mpi_handle.comm, dst_coords, &dst_rank);
      
      // Get local start index to send
      int start_l = MAX(send_start_g - this_proc_start_g, 0);
      int end_l = MIN(send_end_g - this_proc_start_g, this_proc_reduced_size - 1);
      int count = end_l - start_l + 1;
      
      // Pack data into send buffer and send
      if(usedFirstSndBuf) {
        sendReducedMPI(start_l, count, dst_rank, a, c, d, sndbuf_2);
      } else {
        sendReducedMPI(start_l, count, dst_rank, a, c, d, sndbuf_1);
        usedFirstSndBuf = true;
      }
    }
  }
  
  // Allocate receive buffer
  REAL *rcvbuf = (REAL *) malloc(3 * this_proc_reduced_size * numTrids * sizeof(REAL));
  
  // Global start and end index of data needing to receive
  int rcv_start_g = this_proc_start_g - s;
  int rcv_end_g = rcv_start_g + numTrids * this_proc_reduced_size;
  
  // Only need to check procs that are 'below' the current MPI proc for "-s" elements
  for(int i = 0; i <  mpi_handle.coords[solvedim]; i++) {
    int current_start_g = reducedStart_g[i];
    int current_end_g = current_start_g + reducedSizes[i];
     
    // Check if need to receive data from this MPI proc
    if((rcv_start_g <= current_end_g && rcv_start_g >= current_start_g)
        || (rcv_end_g <= current_end_g && rcv_end_g >= current_start_g)
        || (rcv_start_g < current_start_g && rcv_end_g > current_end_g)) {
      // Get rank of MPI proc to receive from
      // Convert source coordinates of MPI node into the node's rank
      int src_coords[3];
      src_coords[0] = mpi_handle.coords[0];
      src_coords[1] = mpi_handle.coords[1];
      src_coords[2] = mpi_handle.coords[2];
      src_coords[solvedim] = i;
      int src_rank = 0;
      MPI_Cart_rank(mpi_handle.comm, src_coords, &src_rank);
      
      // Get local start index to receive to
      int start_l = MAX(current_start_g - rcv_start_g, 0);
      int end_l = MIN(current_end_g - rcv_start_g - 1, this_proc_reduced_size - 1);
      int count = end_l - start_l + 1;
      
      // Receive data and copy to arrays
      receiveReducedMPI(start_l, count, src_rank, a_s, c_s, d_s, rcvbuf);
    }
  }
  
  /*
   * =====================================
   * +S ELEMENTS
   * =====================================
   */
  usedFirstSndBuf = false;
  
  // Only need to check procs that are 'below' the current MPI proc for "+s" elements
  for(int i = 0; i < mpi_handle.coords[solvedim]; i++) {
    int send_start_g = reducedStart_g[i] + s;
    int send_end_g = send_start_g + reducedSizes[i];
     
    // Check if need to send data to this MPI proc
    if((send_start_g <= this_proc_end_g && send_start_g >= this_proc_start_g)
        || (send_end_g <= this_proc_end_g && send_end_g >= this_proc_start_g)
        || (send_start_g < this_proc_start_g && send_end_g > this_proc_end_g)) {
      // Get rank of MPI proc to send to
      // Convert destination coordinates of MPI node into the node's rank
      int dst_coords[3];
      dst_coords[0] = mpi_handle.coords[0];
      dst_coords[1] = mpi_handle.coords[1];
      dst_coords[2] = mpi_handle.coords[2];
      dst_coords[solvedim] = i;
      int dst_rank = 0;
      MPI_Cart_rank(mpi_handle.comm, dst_coords, &dst_rank);
      
      // Get local start index to send
      int start_l = MAX(send_start_g - this_proc_start_g, 0);
      int end_l = MIN(send_end_g - this_proc_start_g, this_proc_reduced_size - 1);
      int count = end_l - start_l + 1;
      
      // Pack data into send buffer and send
      if(usedFirstSndBuf) {
        sendReducedMPI(start_l, count, dst_rank, a, c, d, sndbuf_2);
      } else {
        sendReducedMPI(start_l, count, dst_rank, a, c, d, sndbuf_1);
        usedFirstSndBuf = true;
      }
    }
  }
  
  // The start and end indices for the "+s" elements to recieve
  rcv_start_g = this_proc_start_g + s;
  rcv_end_g = rcv_start_g + numTrids * this_proc_reduced_size;
  
  // Only need to check procs that are 'above' the current MPI proc for "+s" elements
  for(int i = mpi_handle.coords[solvedim] + 1; i <  mpi_handle.pdims[solvedim]; i++) {
    int current_start_g = reducedStart_g[i];
    int current_end_g = current_start_g + reducedSizes[i];
     
    // Check if need to receive data from this MPI proc
    if((rcv_start_g <= current_end_g && rcv_start_g >= current_start_g)
        || (rcv_end_g <= current_end_g && rcv_end_g >= current_start_g)
        || (rcv_start_g < current_start_g && rcv_end_g > current_end_g)) {
      // Get rank of MPI proc to receive from
      // Convert source coordinates of MPI node into the node's rank
      int src_coords[3];
      src_coords[0] = mpi_handle.coords[0];
      src_coords[1] = mpi_handle.coords[1];
      src_coords[2] = mpi_handle.coords[2];
      src_coords[solvedim] = i;
      int src_rank = 0;
      MPI_Cart_rank(mpi_handle.comm, src_coords, &src_rank);
      
      // Get local start index to receive to
      int start_l = MAX(current_start_g - rcv_start_g, 0);
      int end_l = MIN(current_end_g - rcv_start_g - 1, this_proc_reduced_size - 1);
      int count = end_l - start_l + 1;
      
      // Receive data and copy to arrays
      receiveReducedMPI(this_proc_reduced_size + start_l, count, src_rank, a_s, c_s, d_s, rcvbuf);
    }
  }
  
  // Check if need to copy local memory to 'a_s', 'c_s' and 'd_s'
  // Check for -S elements
  rcv_start_g = this_proc_start_g - s;
  rcv_end_g = rcv_start_g + numTrids * this_proc_reduced_size;
  
  if((rcv_start_g <= this_proc_end_g && rcv_start_g >= this_proc_start_g)
      || (rcv_end_g <= this_proc_end_g && rcv_end_g >= this_proc_start_g)
      || (rcv_start_g < this_proc_start_g && rcv_end_g > this_proc_end_g)) {
    
    int snd_start_l = MAX(rcv_start_g - this_proc_start_g, 0);
    
    int rcv_start_l = MAX(this_proc_start_g - rcv_start_g, 0);
    int rcv_end_l = MIN(this_proc_end_g - rcv_start_g - 1, this_proc_reduced_size - 1);
    int count = end_l - start_l + 1;
    
    snd_start_l *= numTrids;
    rcv_start_l *= numTrids;
    count *= numTrids;
  
    cudaMemcpy(&a_s[rcv_start_l], &a[snd_start_l], count * sizeof(REAL), cudaMemcpyDeviceToDevice);
    cudaMemcpy(&c_s[rcv_start_l], &c[snd_start_l], count * sizeof(REAL), cudaMemcpyDeviceToDevice);
    cudaMemcpy(&d_s[rcv_start_l], &d[snd_start_l], count * sizeof(REAL), cudaMemcpyDeviceToDevice);
  }
  
  // Check for +S elements
  rcv_start_g = this_proc_start_g + s;
  rcv_end_g = rcv_start_g + numTrids * this_proc_reduced_size;
  
  if((rcv_start_g <= this_proc_end_g && rcv_start_g >= this_proc_start_g)
      || (rcv_end_g <= this_proc_end_g && rcv_end_g >= this_proc_start_g)
      || (rcv_start_g < this_proc_start_g && rcv_end_g > this_proc_end_g)) {
    
    int snd_start_l = MAX(rcv_start_g - this_proc_start_g, 0);
    
    int rcv_start_l = MAX(this_proc_start_g - rcv_start_g, 0);
    int rcv_end_l = MIN(this_proc_end_g - rcv_start_g - 1, this_proc_reduced_size - 1);
    int count = end_l - start_l + 1;
    
    snd_start_l *= numTrids;
    rcv_start_l += this_proc_reduced_size;
    rcv_start_l *= numTrids;
    count *= numTrids;
  
    cudaMemcpy(&a_s[rcv_start_l], &a[snd_start_l], count * sizeof(REAL), cudaMemcpyDeviceToDevice);
    cudaMemcpy(&c_s[rcv_start_l], &c[snd_start_l], count * sizeof(REAL), cudaMemcpyDeviceToDevice);
    cudaMemcpy(&d_s[rcv_start_l], &d[snd_start_l], count * sizeof(REAL), cudaMemcpyDeviceToDevice);
  }
  
  // Check if need to zero part of 'a_s', 'c_s' and 'd_s'
  // Check for -S elements
  rcv_start_g = this_proc_start_g - s;
  rcv_end_g = rcv_start_g + numTrids * this_proc_reduced_size;
  if(rcv_start_g < 0) {
    int count = MIN(abs(rcv_start_g), this_proc_reduced_size);
    int numThreads = count * numTrids;
    
    int nThreads = 512;
    int nBlocks = 1;
    if(numThreads < 512) {
      nThreads = totalThreads;
    } else {
      nBlocks = (int)ceil((double)totalThreads / (double)nThreads);
    }
    zeroArray<<<nBlocks,nThreads>>>zeroArray(0, count * numTrids, a_s);
    zeroArray<<<nBlocks,nThreads>>>zeroArray(0, count * numTrids, c_s);
    zeroArray<<<nBlocks,nThreads>>>zeroArray(0, count * numTrids, d_s);
  }
  
  // Check for +S elements
  rcv_start_g = this_proc_start_g + s;
  rcv_end_g = rcv_start_g + numTrids * this_proc_reduced_size;
  
  int reduced_end = reducedStart_g[mpi_handle.pdims[solvedim] - 1] + reducedSizes[mpi_handle.pdims[solvedim] - 1] - 1;
  
  if(rcv_end_g > reduced_end) {
    int count = rcv_end_g - reduced_end;
    int start = (2 * this_proc_reduced_size - count) * numTrids;
    
    int numThreads = count * numTrids;
    
    int nThreads = 512;
    int nBlocks = 1;
    if(numThreads < 512) {
      nThreads = totalThreads;
    } else {
      nBlocks = (int)ceil((double)totalThreads / (double)nThreads);
    }
    zeroArray<<<nBlocks,nThreads>>>zeroArray(start, count * numTrids, a_s);
    zeroArray<<<nBlocks,nThreads>>>zeroArray(start, count * numTrids, c_s);
    zeroArray<<<nBlocks,nThreads>>>zeroArray(start, count * numTrids, d_s);
  }
  
  // Free buffers
  free(sndbuf_1);
  free(sndbuf_2);
  free(rcvbuf);
}

// TODO see if more efficient to zero array in chunks
template<typename REAL>
__global__ void zeroArray(int start, int count, REAL* __restrict__ array) {
  int threadId_g = (blockIdx.x * nThreads) + threadIdx.x;
  if(threadId_g <= count) {
    array[start + threadId_g] = (REAL)0.0;
  }
}

template<typename REAL>
void sendReducedMPI(int start, int count, int rank, const REAL* __restrict__ a, 
                    const REAL* __restrict__ c, const REAL* __restrict__ d, REAL* sndbuf) {
  int array_start = start * numTrids;
  int array_count = count * numTrids;
  
  // Copy necessary data to send buffer
  cudaMemcpy(&sndbuf[0], &a[array_start], array_count * sizeof(REAL), cudaMemcpyDeviceToHost);
  cudaMemcpy(&sndbuf[array_count], &c[array_start], array_count * sizeof(REAL), 
             cudaMemcpyDeviceToHost);
  cudaMemcpy(&sndbuf[2 * array_count], &d[array_start], array_count * sizeof(REAL), 
             cudaMemcpyDeviceToHost);
  
  // Send data over MPI
  if(std::is_same<REAL, float>::value) {
    MPI_Send(&sndbuf[0], 3 * array_count, MPI_FLOAT, rank, 0, mpi_handle.comm);
  } else {
    MPI_Send(&sndbuf[0], 3 * array_count, MPI_DOUBLE, rank, 0, mpi_handle.comm);
  }
}

template<typename REAL>
void receiveReducedMPI(int start, int count, int rank, const REAL* __restrict__ a, 
                       const REAL* __restrict__ c, const REAL* __restrict__ d, REAL* rcvbuf) {
  int array_start = start * numTrids;
  int array_count = count * numTrids;
  
  // Receive data from MPI
  if(std::is_same<REAL, float>::value) {
    MPI_Recv(&rcvbuf[0], 3 * array_count, MPI_FLOAT, rank, 0, mpi_handle.comm, 
             MPI_STATUS_IGNORE);
  } else {
    MPI_Recv(&rcvbuf[0], 3 * array_count, MPI_DOUBLE, rank, 0, mpi_handle.comm, 
             MPI_STATUS_IGNORE);
  }
  
  // Copy data to GPU
  cudaMemcpy(&a[array_start], &rcvbuf[0], array_count * sizeof(REAL), cudaMemcpyHostToDevice);
  cudaMemcpy(&c[array_start], &rcvbuf[array_count], array_count * sizeof(REAL), 
             cudaMemcpyHostToDevice);
  cudaMemcpy(&c[array_start], &rcvbuf[2 * array_count], array_count * sizeof(REAL), 
             cudaMemcpyHostToDevice);
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
  int threadId_l = (threadId_g - (tridiag * threadsPerTrid));
  int startElement = threadId_l * regStoreSize;
  
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
    aa_r[tridiag + threadId_l * 2 * numTrids] = aa_reg[0];
    aa_r[tridiag + (threadId_l * 2 + 1) * numTrids] = aa_reg[n];
    cc_r[tridiag + threadId_l * 2 * numTrids] = aa_reg[0];
    cc_r[tridiag + (threadId_l * 2 + 1) * numTrids] = aa_reg[n];
    dd_r[tridiag + threadId_l * 2 * numTrids] = aa_reg[0];
    dd_r[tridiag + (threadId_l * 2 + 1) * numTrids] = aa_reg[n];
    /*aa_r[2 * tridiag]     = aa_reg[0];
    aa_r[2 * tridiag + 1] = aa_reg[n];
    cc_r[2 * tridiag]     = cc_reg[0];
    cc_r[2 * tridiag + 1] = cc_reg[n];
    dd_r[2 * tridiag]     = dd_reg[0];
    dd_r[2 * tridiag + 1] = dd_reg[n];*/
  }
  
  // Store aa, cc and dd values
  storeStridedDataFromRegisters<REAL, regStoreSize>(aa_reg, aa, tridiag, startElement, length, 
                                                    numTrids, stride, batchSize, batchStride);
  
  storeStridedDataFromRegisters<REAL, regStoreSize>(cc_reg, cc, tridiag, startElement, length, 
                                                    numTrids, stride, batchSize, batchStride);
  
  storeStridedDataFromRegisters<REAL, regStoreSize>(dd_reg, dd, tridiag, startElement, length, 
                                                    numTrids, stride, batchSize, batchStride);
}

// Will probably have to call each iteration separately as doubt you can make 
// MPI calls in CUDA
template <typename REAL, int reducedSize, int solvedim, int nBlocks, int nThreads>
void batched_trid_reduced(const REAL* __restrict__ aa_r, const REAL* __restrict__ cc_r, 
                          REAL* __restrict__ dd_r, trid_mpi_handle &mpi_handle) {
  // TODO see if a way of getting this without MPI reduce
  int reducedSize_g;
  MPI_Allreduce(&reducedSize, &reducedSize_g, 1, MPI_INT, MPI_SUM, mpi_handle.y_comm);
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
  int wholeTridThreads = (512 / threadsPerTrid) * 512;
  int wholeTridBlocks = (int)ceil((double)(threadsPerTrid * numTrids) / (double)wholeTridThreads);
  
  // Send and receive initial values
  getInitialValuesForPCR(aa_r, cc_r, dd_r, aa_r_s, cc_r_s, dd_r_s, solvedim, mpi_handle);
  
  // Perform initial step of PCR
  batched_trid_reduced_init_kernel<<<wholeTridBlocks, wholeTridThreads>>>(aa_r, cc_r, dd_r, aa_r_s, cc_r_s, 
                                                          dd_r_s);
  
  for(int p = 1; p <= P; p++) {
    // s = 2^p
    int s = 1 << p;
    
    // Send and receive necessary values
    getValuesForPCR<REAL>(aa_r, cc_r, dd_r, aa_r_s, cc_r_s, dd_r_s, solvedim, mpi_handle);
    
    // Run PCR step on GPU
    batched_trid_reduced_kernel<<<nBlocks, nThreads>>>(aa_r, cc_r, dd_r, aa_r_s, cc_r_s, dd_r_s);
  }
  
  // Communicate boundary values for final step of PCR
  getFinalValuesForPCR(dd_r, dd_r_s, solvedim, mpi_handle);
  
  // Final part of PCR
  batched_trid_reduced_final_kernel<<<wholeTridBlocks, wholeTridThreads>>>(aa_r, cc_r, dd_r, dd_r_s);
  
  // Free memory
  cudaFree(aa_r_s);
  cudaFree(cc_r_s);
  cudaFree(dd_r_s);
}

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
  
  // TODO Copy memory from Host to GPU
  
  if(solvedim == 0) {
    // For x dim might need to transpose (see single node version)
    // Or just write a version for contigous memory
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
    
    int reducedSize = threadsPerTrid * 2;
    // TODO change to one interwoven array once algorithm is working
    REAL *aa_r = NULL;
    REAL *cc_r = NULL;
    REAL *dd_r = NULL;
    cudaMalloc(&aa_r, sizeof(REAL) * reducedSize * numTrids);
    cudaMalloc(&cc_r, sizeof(REAL) * reducedSize * numTrids);
    cudaMalloc(&dd_r, sizeof(REAL) * reducedSize * numTrids);
    
    batched_trid_forwards_kernel<REAL, regStoreSize, tridSolveSize><<<nBlocks, nThreads>>>(
                                      handle.a, handle.b, handle.c, handle.du, aa, cc, dd, aa_r, 
                                      cc_r, dd_r, length, stride, numTrids, batchSize, 
                                      batchStride);
    
    // Call PCR reduced (modified to include MPI comm as reduced system will 
    // be spread over nodes)
    batched_trid_reduced<REAL, reducedSize, solvedim, nBlocks, nThreads>(aa_r, cc_r, dd_r, mpi_handle);
    
    // Call backwards pass
    
    // Free memory
    cudaFree(aa_r);
    cudaFree(cc_r);
    cudaFree(dd_r);
  } else if(solvedim == 2) {
    /*
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
    */
  }
}

#endif
