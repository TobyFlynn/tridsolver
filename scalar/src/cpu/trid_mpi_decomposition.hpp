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
    MPI_Type_create_subarray(ndims, sizes, subsizes, substarts, MPI_ORDER_C, datatype, &subarrays[p]);
    MPI_Type_commit(&subarrays[p]);
  }
}
