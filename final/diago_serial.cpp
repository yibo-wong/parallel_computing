#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <lapacke.h>
#include <sstream>
#include <string>
using namespace std;

void input(double *A, int &n) {
  ifstream file;
  file.open("./result/hamilton.txt");
  if (!file.is_open()) {
    cout << "fail to open!" << endl;
    return;
  }
  int n_size;
  file >> n_size;
  n = n_size;
  for (int i = 0; i < n * n; i++) {
    file >> A[i];
  }
}

int main(int argc, char **argv) {
  int n = 0;
  double *mat = new double[2550];
  double *wr = new double[52];
  double *wi = new double[52];
  double *vl = new double[2550];
  double *vr = new double[2550];

  input(mat, n);
  int lda = n;
  int ldvl = n;
  int ldvr = n;
  int info = 0;
  int lwork = 0;
  char jobvl = 'V';
  char jobvr = 'V';
  LAPACKE_dgeev(LAPACK_ROW_MAJOR, jobvl, jobvr, n, mat, lda, wr, wi, vl, ldvl,
                vr, ldvr);
  ofstream out;
  out.open("./result/diago_lapack.txt");
  out << "-------------------eigenvalue:-------------------" << endl;
  for (int i = 0; i < n; i++) {
    out << wr[i] << endl;
  }
  out << "-------------------eigenvector:-------------------" << endl;
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      out << setw(15) << vr[i * n + j];
    }
    out << endl;
  }
  out.close();
  cout << "The result is printed in file \"./result/diago_lapack.txt\"  . "
          "Check it out!"
       << endl;
  return 0;
}