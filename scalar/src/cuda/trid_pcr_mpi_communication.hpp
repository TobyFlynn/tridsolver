#ifndef TRID_GPU_MPI_PCR_COMM__
#define TRID_GPU_MPI_PCR_COMM__

#include <cmath>
#include <type_traits>

#define ROUND_DOWN(N,step) (((N)/(step))*step)
#define MIN(X,Y) ((X) < (Y) ? (X) : (Y))
#define MAX(X,Y) ((X) > (Y) ? (X) : (Y))

// TODO see if more efficient to zero array in chunks
template<typename REAL>
__global__ void zeroArray(int start, int count, REAL* __restrict__ array) {
  int threadId_g = (blockIdx.x * blockDim.x) + threadIdx.x;
  if(threadId_g <= count) {
    array[start + threadId_g] = (REAL)0.0;
  }
}

template<typename REAL>
void sendReducedMPI(int start, int count, int numTrids, int rank, const REAL* __restrict__ a, 
                    const REAL* __restrict__ c, const REAL* __restrict__ d, REAL* sndbuf, trid_mpi_handle &mpi_handle) {
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
void receiveReducedMPI(int start, int count, int numTrids, int rank, REAL* __restrict__ a, 
                       REAL* __restrict__ c, REAL* __restrict__ d, REAL* rcvbuf, trid_mpi_handle &mpi_handle) {
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

template<typename REAL>
void getInitialValuesForPCR(const REAL* __restrict__ a, const REAL* __restrict__ c, 
                            const REAL* __restrict__ d, REAL* __restrict__ a_s, 
                            REAL* __restrict__ c_s, REAL* __restrict__ d_s, int solvedim, 
                            int numTrids, int threadsPerTrid, int reducedSize, trid_mpi_handle &mpi_handle) {
  // Buffer for a0, an, c0, cn, d0 and dn values for each trid system
  /*
   * sndbuf = | all 'a_0's | all 'c_0's | all 'd_0's | all 'a_n's | all 'c_n's | all 'd_n's |
   */
  REAL *sndbuf = (REAL *) malloc(3 * 2 * numTrids * sizeof(REAL));
  REAL *rcvbuf = (REAL *) calloc(3 * 2 * numTrids, sizeof(REAL));
  
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
  if(solvedim == 0) {
    if(mpi_handle.coords[0] > 0) {
      // Convert destination coordinates of MPI node into the node's rank
      int dst_coords[3];
      dst_coords[0] = mpi_handle.coords[0] - 1;
      dst_coords[1] = mpi_handle.coords[1];
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
  } else if(solvedim == 1) {
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
  if(solvedim == 0) {
    if(mpi_handle.coords[0] < mpi_handle.pdims[0] - 1) {
      // Convert src coordinates of MPI node into the node's rank
      int src_coords[3];
      src_coords[0] = mpi_handle.coords[0] + 1;
      src_coords[1] = mpi_handle.coords[1];
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
  } else if(solvedim == 1) {
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
  if(solvedim == 0) {
    if(mpi_handle.coords[0] < mpi_handle.pdims[0] - 1) {
      // Convert destination coordinates of MPI node into the node's rank
      int dst_coords[3];
      dst_coords[0] = mpi_handle.coords[0] + 1;
      dst_coords[1] = mpi_handle.coords[1];
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
  } else if(solvedim == 1) {
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
  if(solvedim == 0) {
    if(mpi_handle.coords[0] > 0) {
      // Convert src coordinates of MPI node into the node's rank
      int src_coords[3];
      src_coords[0] = mpi_handle.coords[0] - 1;
      src_coords[1] = mpi_handle.coords[1];
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
  } else if(solvedim == 1) {
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

template<typename REAL, int regStoreSize>
void getValuesForPCR(const REAL* __restrict__ a, const REAL* __restrict__ c, 
                     const REAL* __restrict__ d, REAL* __restrict__ a_s, REAL* __restrict__ c_s, 
                     REAL* __restrict__ d_s, int solvedim, int numTrids, int size_g, 
                     int s, trid_mpi_handle &mpi_handle) {
  // Get sizes for each proc and the size of the reduced system
  int numProcs = mpi_handle.pdims[solvedim];
  int tmp = size_g / numProcs;
  int remainder = size_g % numProcs;
  int sizes[numProcs];
  int reducedSizes[numProcs];
  for(int i = 0; i < numProcs; i++) {
    if(i < remainder) {
      sizes[i] = tmp + 1;
    } else {
      sizes[i] = tmp;
    }
    reducedSizes[i] = 2 * (int)ceil((double)sizes[i] / (double)regStoreSize);
  }
  
  // Get global start indices for each proc
  int reducedStart_g[numProcs];
  int total = 0;
  for(int i = 0; i < numProcs; i++) {
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
  for(int i = mpi_handle.coords[solvedim] + 1; i < numProcs; i++) {
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
        sendReducedMPI<REAL>(start_l, count, numTrids, dst_rank, a, c, d, sndbuf_2, mpi_handle);
      } else {
        sendReducedMPI<REAL>(start_l, count, numTrids, dst_rank, a, c, d, sndbuf_1, mpi_handle);
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
      receiveReducedMPI<REAL>(start_l, count, numTrids, src_rank, a_s, c_s, d_s, rcvbuf, mpi_handle);
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
        sendReducedMPI<REAL>(start_l, count, numTrids, dst_rank, a, c, d, sndbuf_2, mpi_handle);
      } else {
        sendReducedMPI<REAL>(start_l, count, numTrids, dst_rank, a, c, d, sndbuf_1, mpi_handle);
        usedFirstSndBuf = true;
      }
    }
  }
  
  // The start and end indices for the "+s" elements to recieve
  rcv_start_g = this_proc_start_g + s;
  rcv_end_g = rcv_start_g + numTrids * this_proc_reduced_size;
  
  // Only need to check procs that are 'above' the current MPI proc for "+s" elements
  for(int i = mpi_handle.coords[solvedim] + 1; i <  numProcs; i++) {
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
      receiveReducedMPI<REAL>(this_proc_reduced_size + start_l, count, numTrids, src_rank, a_s, c_s, d_s, rcvbuf, mpi_handle);
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
    int count = rcv_end_l - rcv_start_l + 1;
    
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
    int count = rcv_end_l - rcv_start_l + 1;
    
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
      nThreads = numThreads;
    } else {
      nBlocks = (int)ceil((double)numThreads / (double)nThreads);
    }
    zeroArray<REAL><<<nBlocks,nThreads>>>(0, count * numTrids, a_s);
    zeroArray<REAL><<<nBlocks,nThreads>>>(0, count * numTrids, c_s);
    zeroArray<REAL><<<nBlocks,nThreads>>>(0, count * numTrids, d_s);
  }
  
  // Check for +S elements
  rcv_start_g = this_proc_start_g + s;
  rcv_end_g = rcv_start_g + numTrids * this_proc_reduced_size;
  
  int reduced_end = reducedStart_g[numProcs - 1] + reducedSizes[numProcs - 1] - 1;
  
  if(rcv_end_g > reduced_end) {
    int count = rcv_end_g - reduced_end;
    int start = (2 * this_proc_reduced_size - count) * numTrids;
    
    int numThreads = count * numTrids;
    
    int nThreads = 512;
    int nBlocks = 1;
    if(numThreads < 512) {
      nThreads = numThreads;
    } else {
      nBlocks = (int)ceil((double)numThreads / (double)nThreads);
    }
    zeroArray<REAL><<<nBlocks,nThreads>>>(start, count * numTrids, a_s);
    zeroArray<REAL><<<nBlocks,nThreads>>>(start, count * numTrids, c_s);
    zeroArray<REAL><<<nBlocks,nThreads>>>(start, count * numTrids, d_s);
  }
  
  // Free buffers
  free(sndbuf_1);
  free(sndbuf_2);
  free(rcvbuf);
}

template<typename REAL>
void getFinalValuesForPCR(const REAL* __restrict__ d, REAL* __restrict__ d_s,
                          int solvedim, int numTrids, trid_mpi_handle &mpi_handle) {
  REAL *sndbuf = (REAL *) malloc(numTrids * sizeof(REAL));
  REAL *rcvbuf = (REAL *) calloc(numTrids, sizeof(REAL));
  
  cudaMemcpy(&sndbuf[0], &d[0], numTrids * sizeof(REAL), cudaMemcpyDeviceToHost);
  
  // Send
  if(solvedim == 0) {
    if(mpi_handle.coords[0] > 0) {
      // Convert destination coordinates of MPI node into the node's rank
      int dst_coords[3];
      dst_coords[0] = mpi_handle.coords[0] - 1;
      dst_coords[1] = mpi_handle.coords[1];
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
  } else if(solvedim == 1) {
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
  if(solvedim == 0) {
    if(mpi_handle.coords[0] < mpi_handle.pdims[0] - 1) {
      // Convert src coordinates of MPI node into the node's rank
      int src_coords[3];
      src_coords[0] = mpi_handle.coords[0] + 1;
      src_coords[1] = mpi_handle.coords[1];
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
  } else if(solvedim == 1) {
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
  cudaMemcpy(&d_s[0], &rcvbuf[0], numTrids * sizeof(REAL), cudaMemcpyHostToDevice);
}

#endif
