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

class Matrix_float {
private:
  int ncols = 0;
  int nrows = 0;
  float *element;

public:
  // A naive construction function. All elements are set zero.
  Matrix_float() { element = NULL; }
  Matrix_float(int nr, int nc) {
    if (nr <= 0 || nc <= 0) {
      cout << "Rows or cols cannot be zero!" << endl;
      nrows = ncols = 0;
      element = NULL;
    } else {
      nrows = nr;
      ncols = nc;
      element = new float[nrows * ncols];
      for (int i = 0; i < nrows * ncols; i++)
        element[i] = 0;
    }
  }
  int getRows() const { return nrows; }
  int getCols() const { return ncols; }
  // overload operator [], so that i can get the value i,j by using
  // "Matrix[i][j]"

  float &operator()(const int &x, const int &y) const {
    return element[x * ncols + y];
  }
  // overload *= to calculate k*A
  void operator*=(const float &value) {
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
      element = new float[nrows * ncols];
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
  float maxElement() {
    float result = -(1 << 30);
    for (int i = 0; i < nrows; i++)
      for (int j = 0; j < ncols; j++)
        result = max(result, element[i * ncols + j]);
    return result;
  }
  // calculate the min element
  float minElement() {
    float result = +(1 << 30);
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
  int init(float *ele) {
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

  friend int RSMdiago(Matrix_float &A, Matrix_float &wr, Matrix_float &wi,
                      Matrix_float &vr, Matrix_float &vl) {
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

    LAPACKE_sgeev(LAPACK_ROW_MAJOR, jobvl, jobvr, n, A.element, lda, wr.element,
                  wi.element, vl.element, ldvl, vr.element, ldvr);

    return info;
  }

  ~Matrix_float() {}

  friend ostream &operator<<(ostream &out, Matrix_float &A) {
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
  // need rewrite!!!!!
  friend Matrix_float blas_matmul(Matrix_float &A, Matrix_float &B) {
    int m = A.nrows;
    int n = B.ncols;
    int k = A.ncols;
    float alpha = 1.0;
    float beta = 0.0;
    float *matA = A.element;
    float *matB = B.element;
    float *matC = new float[m * n];
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, alpha, matA,
                k, matB, n, beta, matC, n);
    Matrix_float C(m, n);
    C.init(matC);
    return C;
  }
};

Matrix_float operator+(Matrix_float &A, Matrix_float &B) {
  int r = A.getRows();
  int c = A.getCols();
  Matrix_float result = Matrix_float(r, c);
  if (A.getRows() != B.getRows() || A.getCols() != B.getCols()) {
    cout << "not same shape" << endl;
    return result;
  }
  for (int i = 0; i < r; i++)
    for (int j = 0; j < r; j++)
      result(i, j) = A(i, j) + B(i, j);
  return result;
}

Matrix_float operator-(Matrix_float &A, Matrix_float &B) {
  int r = A.getRows();
  int c = A.getCols();
  Matrix_float result = Matrix_float(r, c);
  if (A.getRows() != B.getRows() || A.getCols() != B.getCols()) {
    cout << "not same shape" << endl;
    return result;
  }
  for (int i = 0; i < r; i++)
    for (int j = 0; j < r; j++)
      result(i, j) = A(i, j) - B(i, j);
  return result;
}

Matrix_float operator*(Matrix_float &A, Matrix_float &B) {
  int r = A.getRows();
  int c = B.getCols();
  int p = A.getCols();
  Matrix_float result = Matrix_float(r, c);
  if (A.getCols() != B.getRows()) {
    cout << "wrong dimension" << endl;
    return result;
  }
  for (int i = 0; i < r; i++) {
    for (int j = 0; j < c; j++) {
      float sum = 0;
      for (int k = 0; k < p; k++) {
        sum += A(i, k) * B(k, j);
      }
      result(i, j) = sum;
    }
  }
  return result;
}
