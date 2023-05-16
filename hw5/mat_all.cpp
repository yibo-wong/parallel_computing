// Yibo Wang, 2100011025, coe_pku, parallel_programming.
#include <cblas.h>
#include <chrono>
#include <cmath>
#include <cstring>
#include <input.h>
#include <iomanip>
#include <iostream>
#include <lapacke.h>
#include <map>
#include <matrix.h>
#include <string>
#include <timer.h>
#include <vector>

int main() {
  Input_demand *input_demand = new Input_demand();
  ifstream demand_file;
  demand_file.open("demand.txt");
  input_demand->read_in(demand_file);
  demand_file.close();

  if (input_demand->calculation == "matmul") {
    Matrix A, B;
    if (input_demand->matrix_type == "file") {
      Input *input = new Input();
      ifstream file;
      file.open(input_demand->matrix_1_name);
      input->read_in(file);
      file.close();

      A.resize(input->nrows, input->ncols);
      A.init(input->element);

      file.open(input_demand->matrix_2_name);
      input->read_in(file);
      file.close();
      B.resize(input->nrows, input->ncols);
      B.init(input->element);
    }
    timer::tick("self", "made");
    Matrix C = A * B;
    timer::tick("self", "made");

    timer::tick("cblas", "dgemm");
    Matrix D = blas_matmul(A, B);
    timer::tick("cblas", "dgemm");

    if (input_demand->timer_print)
      timer::print();
    if (input_demand->result_print) {
      ofstream output;
      output.open("result.txt");
      output << "--------self-made-matrix-multiplication--------" << endl;
      output << setprecision(15) << C << endl;
      output << "------------cblas-dgemm-multuplication---------" << endl;
      output << setprecision(15) << D << endl;
      output.close();
      cout << endl << "The result is in result.txt. Check it out!" << endl;
    }
  } else if (input_demand->calculation == "rsmdiago") {
    Matrix A;
    if (input_demand->matrix_type == "file") {
      Input *input = new Input();
      ifstream file;
      file.open(input_demand->matrix_1_name);
      cout << input_demand->matrix_1_name << endl;
      input->read_in(file);
      file.close();
      A.resize(input->nrows, input->ncols);
      A.init(input->element);

      Matrix real_eigen_value(1, A.getRows());
      Matrix imag_eigen_value(1, A.getRows());
      Matrix left_eigen_vector(A.getRows(), A.getRows());
      Matrix right_eigen_vector(A.getRows(), A.getRows());
      timer::tick("LAPACK", "dgeev");
      RSMdiago(A, real_eigen_value, imag_eigen_value, left_eigen_vector,
               right_eigen_vector);
      timer::tick("LAPACK", "dgeev");

      if (input_demand->timer_print)
        timer::print();
      if (input_demand->result_print) {
        ofstream output;
        output.open("result.txt");
        output << "--------eigen_value---------" << endl;
        output << real_eigen_value << endl;
        output << "--------eigen_vector--------" << endl;
        output << right_eigen_vector << endl;
        output.close();
        cout << endl << "The result is in result.txt. Check it out!" << endl;
      }
    }
  }
  return 0;
}
/*
g++ mat_all.cpp -o mat_all -llapacke -llapack -lcblas  -lrefblas  -lm -lgfortran
*/