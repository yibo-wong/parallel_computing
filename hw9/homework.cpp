// Yibo Wang, 2100011025, coe_pku, parallel_programming.
#include "./include/input.h"
#include "./include/matrix.h"
#include "./include/timer.h"
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#define max_string_length 100

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
  MPI_Status status;
#endif
  Matrix mat_a, mat_b, mat_result;

  int print_log = 0;
  int timer_print = 0;
  string file_a = "";
  string file_b = "";
  string output = "";
  char *output_c = new char[max_string_length];

  int my_start = 0;
  int row_num = 0;
  int col_num = 0;
  int total_rows = 0;
  double *send_data_a = new double[5000];
  double *send_data_b = new double[5000];
  double alpha = 0, beta = 0;
  Matrix final_result;

  if (world_rank == 0) {
    // read in input.txt

    timer::tick("main: read", "matrix");

    Input_demand *input_demand = new Input_demand();
    ifstream file;
    file.open("INPUT.txt");
    input_demand->read_in(file);
    file.close();
    // assign the values
    output = input_demand->output_to_file;
    for (int i = 0; i < output.size(); i++)
      output_c[i] = output[i];
    print_log = input_demand->result_print;
    timer_print = input_demand->timer_print;
    file_a = input_demand->matrix_1_name;
    file_b = input_demand->matrix_2_name;
    alpha = input_demand->alpha;
    beta = input_demand->beta;
    // read in the two matrix

    ifstream file_2;
    Input *input_b = new Input();
    file_2.open(file_b);
    input_b->read_in(file_2);
    file_2.close();

    ifstream file_1;
    Input *input_a = new Input();
    file_1.open(file_a);
    input_a->read_in(file_1);
    file_1.close();

    timer::tick("main: read", "matrix");

    if (input_a->ncols != input_b->ncols || input_a->nrows != input_b->nrows) {
      cout << "Wrong size: number of rows & cols must be the same." << endl;
      return 1;
    }
    total_rows = input_a->nrows;
    col_num = input_a->ncols;
    final_result = Matrix(total_rows, col_num);
    // broadcast alpha,beta,print_log,timer_print,output_c,total_rows,col_num

    timer::tick("main: send", "data");

#ifdef __MPI
    MPI_Bcast(&alpha, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&beta, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&print_log, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&timer_print, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(output_c, max_string_length, MPI_CHAR, 0, MPI_COMM_WORLD);
    MPI_Bcast(&total_rows, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&col_num, 1, MPI_INT, 0, MPI_COMM_WORLD);
#endif

    timer::tick("main: send", "data");

    if (world_size == 1) {
      row_num = total_rows;
      mat_a = Matrix(total_rows, col_num);
      mat_b = Matrix(total_rows, col_num);
    } else {
      row_num = total_rows / world_size + 1;
      mat_a = Matrix(total_rows / world_size + 1, col_num);
      mat_b = Matrix(total_rows / world_size + 1, col_num);
    }
    mat_a.init(input_a->element, 0);
    mat_b.init(input_b->element, 0);
#ifdef __MPI
    for (int i = 1; i < world_size; i++) {
      int rank = i;
      int ave = total_rows / world_size;
      int this_row_num = (rank < total_rows % world_size) ? ave + 1 : ave;
      int row_start = (rank < total_rows % world_size)
                          ? rank * (ave + 1)
                          : (total_rows % world_size) * (ave + 1) +
                                (rank - total_rows % world_size) * ave;

      send_data_a = input_a->element + row_start * col_num;
      send_data_b = input_b->element + row_start * col_num;
      MPI_Send(&row_start, 1, MPI_INT, rank, 1, MPI_COMM_WORLD);
      MPI_Send(&this_row_num, 1, MPI_INT, rank, 2, MPI_COMM_WORLD);
      MPI_Send(send_data_a, this_row_num * col_num, MPI_DOUBLE, rank, 3,
               MPI_COMM_WORLD);
      MPI_Send(send_data_b, this_row_num * col_num, MPI_DOUBLE, rank, 4,
               MPI_COMM_WORLD);
    }
#endif
  } // in worlds other than 0
  else {

    timer::tick("other: recv", "data");

    // receive lots of stuff
#ifdef __MPI
    MPI_Bcast(&alpha, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&beta, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&print_log, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&timer_print, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(output_c, max_string_length, MPI_CHAR, 0, MPI_COMM_WORLD);
    output = output_c;
    MPI_Bcast(&total_rows, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&col_num, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Recv(&my_start, 1, MPI_INT, 0, 1, MPI_COMM_WORLD, &status);
    MPI_Recv(&row_num, 1, MPI_INT, 0, 2, MPI_COMM_WORLD, &status);
    MPI_Recv(send_data_a, row_num * col_num, MPI_DOUBLE, 0, 3, MPI_COMM_WORLD,
             &status);
    MPI_Recv(send_data_b, row_num * col_num, MPI_DOUBLE, 0, 4, MPI_COMM_WORLD,
             &status);
    mat_a = Matrix(row_num, col_num);
    mat_b = Matrix(row_num, col_num);
    mat_a.init(send_data_a);
    mat_b.init(send_data_b);
#endif
    timer::tick("other: recv", "data");
  }
  // now in all threads

  timer::tick("calc", "add");
  mat_result = (mat_a * alpha) + (mat_b * beta);
  timer::tick("calc", "add");
  if (world_rank == 0)
    system("test -d output || mkdir output");

    // // -----------delete this!!!--------------
    // system("test -d output || mkdir output");
    // stringstream filename;
    // filename << "./output/processor" << world_rank << ".log";
    // string name = filename.str();
    // ofstream out;
    // out.open(name);
    // out << mat_result << endl;
    // out << "-------------" << endl;
    // out.close();
    // // -----------delete this!!!--------------

    // thread 0 collects all the data.
#ifdef __MPI
  if (world_rank != 0) {
    timer::tick("other: send", "data");
    MPI_Send(mat_result.element, row_num * col_num, MPI_DOUBLE, 0, 5,
             MPI_COMM_WORLD);
    timer::tick("other: send", "data");
  } else {
    timer::tick("main: recv", "data");
    final_result.init_partly(mat_result.element, 0, row_num * col_num);
    for (int i = 1; i < world_size; i++) {
      int rank = i;
      int ave = total_rows / world_size;
      int this_row_num = (rank < total_rows % world_size) ? ave + 1 : ave;
      int row_start = (rank < total_rows % world_size)
                          ? rank * (ave + 1)
                          : (total_rows % world_size) * (ave + 1) +
                                (rank - total_rows % world_size) * ave;
      double *recv_data = new double[this_row_num * col_num];
      MPI_Recv(recv_data, this_row_num * col_num, MPI_DOUBLE, i, 5,
               MPI_COMM_WORLD, &status);
      final_result.init_partly(recv_data, row_start * col_num,
                               col_num * this_row_num);
    }
    timer::tick("main: recv", "data");
    timer::tick("main: write", "file");
    ofstream out;
    out.open("./" + output);
    out << final_result << endl;
    out.close();
    timer::tick("main: write", "file");
  }
#else
  final_result = mat_result;
  timer::tick("main: write", "file");
  ofstream out;
  out.open("./" + output);
  out << final_result << endl;
  out.close();
  timer::tick("main: write", "file");
#endif

#ifdef __MPI
  for (int i = 0; i < world_size; i++) {
    if (world_rank == i) {
      cout << endl << "IN WORLD " << world_rank << " :" << endl;
      timer::print();
    }
    MPI_Barrier(MPI_COMM_WORLD);
  }
  MPI_Finalize();
#else
  timer::print();
#endif

  return 0;
}
