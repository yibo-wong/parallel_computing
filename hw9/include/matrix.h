// Yibo Wang, 2100011025, coe_pku, parallel_programming.
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>

using namespace std;

class Matrix {
public:
  int ncols = 0;
  int nrows = 0;
  double *element;

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
  Matrix &operator*=(const double &value) {
    for (int i = 0; i < nrows; i++)
      for (int j = 0; j < ncols; j++)
        element[i * ncols + j] *= value;
    return *this;
  }

  Matrix &operator+=(const double &value) {
    for (int i = 0; i < nrows; i++)
      for (int j = 0; j < ncols; j++)
        element[i * ncols + j] += value;
    return *this;
  }

  Matrix &operator-=(const double &value) {
    for (int i = 0; i < nrows; i++)
      for (int j = 0; j < ncols; j++)
        element[i * ncols + j] -= value;
    return *this;
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
  int init(double *ele, int move = 0) {
    double *pele = ele;
    pele += move;
    for (int i = 0; i < nrows * ncols; i++)
      element[i] = pele[i];
    return 0;
  }

  int init_partly(double *ele, int move = 0, int length = 0) {
    double *pele = ele;
    for (int i = 0; i < length; i++)
      element[i + move] = pele[i];
    return 0;
  }
  ~Matrix() {}

  friend ostream &operator<<(ostream &out, Matrix &A) {
    int r = A.getRows();
    int c = A.getCols();
    for (int i = 0; i < r; i++) {
      for (int j = 0; j < c - 1; j++) {
        out << setprecision(7) << setw(18) << A(i, j) << ", ";
      }
      out << setprecision(7) << setw(18) << A(i, c - 1) << endl;
    }
    return out;
  }
};

Matrix operator+(const Matrix &A, const Matrix &B) {
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

Matrix operator-(const Matrix &A, const Matrix &B) {
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

Matrix operator*(const Matrix &A, const Matrix &B) {
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

Matrix operator*(const Matrix &A, double b) {
  int r = A.getRows();
  int c = A.getCols();
  Matrix result = Matrix(r, c);
  for (int i = 0; i < r; i++)
    for (int j = 0; j < c; j++)
      result(i, j) = A(i, j) * b;
  return result;
}

Matrix operator*(double b, const Matrix A) {
  int r = A.getRows();
  int c = A.getCols();
  Matrix result = Matrix(r, c);
  for (int i = 0; i < r; i++)
    for (int j = 0; j < c; j++)
      result(i, j) = A(i, j) * b;
  return result;
}
