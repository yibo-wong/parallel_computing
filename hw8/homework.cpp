// Yibo Wang, 2100011025, coe_pku, parallel_programming.
#include "input.h"
#include "matrix.h"
#include "timer.h"
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>

#ifdef __MPI
#include <mpi.h>
#endif

using namespace std;

int main(int argc, char **argv) {
  int world_size = 1;
  int world_rank = 0;

  // start the timer
#ifdef __MPI
  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  timer::tick("read_matrix", "(parallel)");
#else
  timer::tick("read_matrix", "(serial)");
#endif

  Matrix my_mat;

  int do_print = 0;
  string dir = "output";
  int my_start = 0;
  int row_num = 0;
  int col_num = 0;
  int total_rows = 0;
  double *send_data = new double[5000];

  if (world_rank == 0) {
    // Matrix send_mat;
    Input *input = new Input();
    ifstream file;
    file.open("test.txt");
    input->read_in(file);
    file.close();

    col_num = input->ncols;
#ifdef __MPI
    cout << col_num << endl;
    MPI_Bcast(&col_num, 1, MPI_INT, 0, MPI_COMM_WORLD);
    cout << "bcast end" << endl;
#endif
    total_rows = input->nrows;

    if (input->print_mpi_log) {
      stringstream ss;
      ss << " test -d " << dir << " || mkdir " << dir;
      system(ss.str().c_str());
      do_print = 1;
#ifdef __MPI
      MPI_Bcast(&do_print, 1, MPI_INT, 0, MPI_COMM_WORLD);
#endif
    }
    if (world_size == 1) {
      row_num = total_rows;
      my_mat = Matrix(total_rows, col_num);
    } else {
      row_num = total_rows / world_size + 1;
      my_mat = Matrix(total_rows / world_size + 1, col_num);
    }
    my_mat.init(input->element, 0);
#ifdef __MPI
    for (int i = 1; i < world_size; i++) {
      int rank = i;
      int ave = total_rows / world_size;
      int row_num = (rank < total_rows % world_size) ? ave + 1 : ave;
      int row_start = (rank < total_rows % world_size)
                          ? rank * (ave + 1)
                          : (total_rows % world_size) * (ave + 1) +
                                (rank - total_rows % world_size) * ave;
      // send_mat = Matrix(row_num, col_num);
      send_data = input->element + row_start * col_num;
      // send_mat.init(input->element, row_start * col_num);
      MPI_Send(&row_start, 1, MPI_INT, rank, 1, MPI_COMM_WORLD);
      MPI_Send(&row_num, 1, MPI_INT, rank, 2, MPI_COMM_WORLD);
      MPI_Send(send_data, row_num * col_num, MPI_DOUBLE, rank, 4,
               MPI_COMM_WORLD);
      cout << row_num * col_num << "out" << endl;
    }
#endif
  } else {
#ifdef __MPI
    MPI_Bcast(&col_num, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&do_print, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Status status;
    MPI_Recv(&my_start, 1, MPI_INT, 0, 1, MPI_COMM_WORLD, &status);
    MPI_Recv(&row_num, 1, MPI_INT, 0, 2, MPI_COMM_WORLD, &status);
    cout << col_num << "in" << endl;
    MPI_Recv(send_data, row_num * col_num, MPI_DOUBLE, 0, 4, MPI_COMM_WORLD,
             &status);
    my_mat = Matrix(row_num, col_num);
    my_mat.init(send_data);
#endif
  }
  if (do_print) {
    stringstream filename;
    filename << "./" << dir << "/processor" << world_rank << ".log";
    string name = filename.str();
    ofstream out;
    out.open(name);
    cout << name << endl;

    // int row_num = my_mat.getRows();

    out << "Process ID: " << world_rank << endl;
    out << "Block Matrix ID: " << world_rank << endl;
    out << "Block Size: " << row_num << " x " << col_num << endl;
    out << "Start Position: (" << my_start << ", 0)" << endl;
    out << "End Position: (" << my_start + row_num << ", " << col_num << ")"
        << endl;
    out << "Block Matrix Elements:" << endl;
    out << my_mat << endl;
  }
#ifdef __MPI
  timer::tick("read_matrix", "(parallel)");
  MPI_Barrier(MPI_COMM_WORLD);
  cout << endl << "IN WORLD " << world_rank << " :" << endl;
  timer::print();
  MPI_Finalize();
#else
  timer::tick("read_matrix", "(serial)");
  timer::print();
#endif

  return 0;
}
