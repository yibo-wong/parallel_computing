#include <algorithm>
#include <cblas.h>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <lapacke.h>
#include <sstream>
#include <string>

using namespace std;

class Matrix {
private:
  int ncols = 0;
  int nrows = 0;
  double *element;

public:
  // A naive construction function. All elements are set zero.
  Matrix() { element = NULL; }
  Matrix(int nr, int nc) {
    if (nr <= 0 || nc <= 0) {
      cout << "Rows or cols cannot be zero!" << endl;
      nrows = ncols = 0;
      element = NULL;
    } else {
      nrows = nr;
      ncols = nc;
      element = new double[nrows * ncols];
      for (int i = 0; i < nrows * ncols; i++)
        element[i] = 0;
    }
  }
  int getRows() const { return nrows; }
  int getCols() const { return ncols; }
  // overload operator [], so that i can get the value i,j by using
  // "Matrix[i][j]"

  double &operator()(const int &x, const int &y) const {
    return element[x * ncols + y];
  }
  // overload *= to calculate k*A
  void operator*=(const double &value) {
    for (int i = 0; i < nrows; i++)
      for (int j = 0; j < ncols; j++)
        element[i * ncols + j] *= value;
  }

  void resize(const int &nr, const int &nc) {
    delete[] element;
    if (nr <= 0 || nc <= 0) {
      cout << "Rows or cols cannot be zero!" << endl;
      nrows = ncols = 0;
      element = NULL;
    } else {
      nrows = nr;
      ncols = nc;
      element = new double[nrows * ncols];
      for (int i = 0; i < nrows * ncols; i++)
        element[i] = 0;
    }
  }

  void operator*=(const int &value) {
    for (int i = 0; i < nrows; i++)
      for (int j = 0; j < ncols; j++)
        element[i * ncols + j] *= value;
  }

  // calculate the max element
  double maxElement() {
    double result = -(1 << 30);
    for (int i = 0; i < nrows; i++)
      for (int j = 0; j < ncols; j++)
        result = max(result, element[i * ncols + j]);
    return result;
  }
  // calculate the min element
  double minElement() {
    double result = +(1 << 30);
    for (int i = 0; i < nrows; i++)
      for (int j = 0; j < ncols; j++)
        result = min(result, element[i * ncols + j]);
    return result;
  }
  // set all elements to 0; return 0 if succeed.
  int toZero() {
    for (int i = 0; i < nrows; i++)
      for (int j = 0; j < ncols; j++)
        element[i * ncols + j] = 0;
    return 0;
  }
  // initialize the matrix. return 0 if succeeds, 1 if fails.
  int init(double *ele) {
    for (int i = 0; i < nrows * ncols; i++)
      element[i] = ele[i];
    return 0;
  }

  bool isRSM() {
    if (nrows != ncols)
      return 0;
    bool flag = 1;
    for (int i = 0; i < nrows; i++) {
      for (int j = i + 1; j < ncols; j++) {
        if (element[i * ncols + j] != element[j * ncols + i])
          return 0;
      }
    }
    return 1;
  }

  friend int RSMdiago(Matrix &A, Matrix &wr, Matrix &wi, Matrix &vr,
                      Matrix &vl) {
    if (!A.isRSM()) {
      return 1;
    }
    int n = A.getRows();
    int lda = n;
    int ldvl = n;
    int ldvr = n;
    int info = 0;
    int lwork = 0;
    char jobvl = 'V';
    char jobvr = 'V';

    LAPACKE_dgeev(LAPACK_ROW_MAJOR, jobvl, jobvr, n, A.element, lda, wr.element,
                  wi.element, vl.element, ldvl, vr.element, ldvr);

    return info;
  }

  ~Matrix() {}

  friend ostream &operator<<(ostream &out, Matrix &A) {
    int r = A.getRows();
    int c = A.getCols();
    for (int i = 0; i < r; i++) {
      for (int j = 0; j < c; j++) {
        out << setprecision(7) << setw(18) << A(i, j) << "  ";
      }
      out << endl;
    }
    return out;
  }

  friend Matrix blas_matmul(Matrix &A, Matrix &B) {
    int m = A.nrows;
    int n = B.ncols;
    int k = A.ncols;
    double alpha = 1.0;
    double beta = 0.0;
    double *matA = A.element;
    double *matB = B.element;
    double *matC = new double[m * n];
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, alpha, matA,
                k, matB, n, beta, matC, n);
    Matrix C(m, n);
    C.init(matC);
    return C;
  }
};

Matrix operator+(Matrix &A, Matrix &B) {
  int r = A.getRows();
  int c = A.getCols();
  Matrix result = Matrix(r, c);
  if (A.getRows() != B.getRows() || A.getCols() != B.getCols()) {
    cout << "not same shape" << endl;
    return result;
  }
  for (int i = 0; i < r; i++)
    for (int j = 0; j < c; j++)
      result(i, j) = A(i, j) + B(i, j);
  return result;
}

Matrix operator-(Matrix &A, Matrix &B) {
  int r = A.getRows();
  int c = A.getCols();
  Matrix result = Matrix(r, c);
  if (A.getRows() != B.getRows() || A.getCols() != B.getCols()) {
    cout << "not same shape" << endl;
    return result;
  }
  for (int i = 0; i < r; i++)
    for (int j = 0; j < c; j++)
      result(i, j) = A(i, j) - B(i, j);
  return result;
}

Matrix operator*(Matrix &A, Matrix &B) {
  int r = A.getRows();
  int c = B.getCols();
  int p = A.getCols();
  Matrix result = Matrix(r, c);
  if (A.getCols() != B.getRows()) {
    cout << "wrong dimension" << endl;
    return result;
  }
  for (int i = 0; i < r; i++) {
    for (int j = 0; j < c; j++) {
      double sum = 0;
      for (int k = 0; k < p; k++) {
        sum += A(i, k) * B(k, j);
      }
      result(i, j) = sum;
    }
  }
  return result;
}
