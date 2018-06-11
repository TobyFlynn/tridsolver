#define CATCH_CONFIG_RUNNER
#include "catch.hpp"

#include <mpi.h>
#include <cstdlib>

int main(int argc, char *argv[])
{
  auto rc = MPI_Init(&argc, &argv);
  if (rc != MPI_SUCCESS) {
    printf("Error starting MPI program. Terminating.\n");
    MPI_Abort(MPI_COMM_WORLD, rc);
  }

  // For the debug prints
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  std::srand(rank);

  int result = Catch::Session().run(argc, argv);

  MPI_Finalize();
  return result;
}
