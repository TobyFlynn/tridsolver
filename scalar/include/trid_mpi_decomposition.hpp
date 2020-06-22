#ifndef __TRID_MPI_PENCIL_DECOMP_HPP
#define __TRID_MPI_PENCIL_DECOMP_HPP

#include "mpi.h"

#define min(x, y) (((x) < (y)) ? (x) : (y))

void decompose(int N, int M, int p, int *n, int *s) {
  int q = N / M;
  int r = N % M;
  *n = q + (r > p);
  *s = q * p + min(r, p);
}

void subarray(MPI_Datatype datatype, int ndims, int sizes[ndims], int axis, int nparts, MPI_Datatype subarrays[nparts]) {
  int subsizes[ndims], substarts[ndims], n, s;
  for(int i = 0; i < ndims; i++) {
    subsizes[i] = sizes[i];
    substarts[i] = 0;
  }
  for(int p = 0; p < nparts; p++) {
    decompose(sizes[axis], nparts, p, &n, &s);
    subsizes[axis] = n;
    substarts[axis] = s;
    MPI_Type_create_subarray(ndims, sizes, subsizes, substarts, MPI_ORDER_FORTRAN, datatype, &subarrays[p]);
    MPI_Type_commit(&subarrays[p]);
  }
}

void exchange(MPI_Comm comm, MPI_Datatype datatype, int ndims, int sizesA[ndims], void *arrayA,
              int axisA, int sizesB[ndims], void *arrayB, int axisB) {
  int nparts;
  MPI_Comm_size(comm, &nparts);
  MPI_Datatype subarraysA[nparts], subarraysB[nparts];

  subarray(datatype, ndims, sizesA, axisA, nparts, subarraysA);
  subarray(datatype, ndims, sizesB, axisB, nparts, subarraysB);

  int counts[nparts], displs[nparts];
  for(int p = 0; p < nparts; p++) {
    counts[p] = 1;
    displs[p] = 0;
  }

  MPI_Alltoallw(arrayA, counts, displs, subarraysA,
                arrayB, counts, displs, subarraysB, comm);

  for(int p = 0; p < nparts; p++) {
    MPI_Type_free(&subarraysA[p]);
    MPI_Type_free(&subarraysB[p]);
  }
}

void subcomm(MPI_Comm comm, int ndims, MPI_Comm subcomms[ndims]) {
  MPI_Comm comm_cart;
  int nprocs, dims[ndims], periods[ndims], remdims[ndims];
  for(int i = 0; i < ndims; i++) {
    dims[i] = periods[i] = remdims[i] = 0;
  }
  MPI_Comm_size(comm, &nprocs);
  MPI_Dims_create(nprocs, ndims, dims);
  MPI_Cart_create(comm, ndims, dims, periods, 1, &comm_cart);
  for(int i = 0; i < ndims; i++) {
    remdims[i] = 1;
    MPI_Cart_sub(comm_cart, remdims, &subcomms[i]);
    remdims[i] = 0;
  }
  MPI_Comm_free(&comm_cart);
}

int lsz(int N, MPI_Comm comm) {
  int size, rank, n, s;
  MPI_Comm_size(comm, &size);
  MPI_Comm_rank(comm, &rank);
  decompose(N, size, rank, &n, &s);
  return n;
}

#endif
