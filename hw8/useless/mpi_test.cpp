#include <cstdio>
#include <iostream>
#include <mpi.h>

using namespace std;

int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);
  int world_size;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  int world_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

  cout << "world size: " << world_size << "  world rank: " << world_rank
       << endl;
  MPI_Finalize();
  return 0;
}
