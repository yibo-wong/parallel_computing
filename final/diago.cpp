#include "./include/scalapack_connector.h"
#include "mpi.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
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
  int myrank, mysize;
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
  MPI_Comm_size(MPI_COMM_WORLD, &mysize);
  MPI_Barrier(MPI_COMM_WORLD);

  int n = 0;

  double *mat;
  mat = new double[2550];

  input(mat, n);

  int nprow = 2;
  int npcol = 2;
  int nb = ceil(double(n) / nprow);
  char jobz = 'V';
  char uplo = 'U';
  char layout = 'R';

  int izero = 0;
  int ione = 1;

  if (nprow * npcol != mysize) {
    cout << "Nproc and nprow*npcol not match!" << endl;
    return 0;
  }
  int mypnum, nprocs;
  int zero = 0;
  int icontxt, myrow, mycol;
  blacs_pinfo_(&mypnum, &nprocs);
  blacs_get_(&zero, &zero, &icontxt);
  blacs_gridinit_(&icontxt, &layout, &nprow, &npcol);
  blacs_gridinfo_(&icontxt, &nprow, &npcol, &myrow, &mycol);

  int ia = myrow * nb;
  int ja = mycol * nb;

  int mpA = numroc_(&n, &nb, &myrow, &izero, &nprow);
  int nqA = numroc_(&n, &nb, &mycol, &izero, &npcol);

  double *A = new double[mpA * nqA];
  double *Z = new double[mpA * nqA];
  double *W = new double[n];
  int k = 0;

  int describeA[9];
  int describeZ[9];
  int info;
  int lddA = max(mpA, 1);
  descinit_(describeA, &n, &n, &nb, &nb, &izero, &izero, &icontxt, &lddA,
            &info);
  descinit_(describeZ, &n, &n, &nb, &nb, &izero, &izero, &icontxt, &lddA,
            &info);

  for (int i = 1; i <= n; i++) {
    for (int j = 1; j <= n; j++) {
      pdelset_(A, &i, &j, describeA, &(mat[(j - 1) + (i - 1) * n]));
    }
  }

  double *work = new double[2];
  int lwork = -1;
  MPI_Barrier(MPI_COMM_WORLD);
  pdsyev_(&jobz, &uplo, &n, A, &ione, &ione, describeA, W, Z, &ione, &ione,
          describeZ, work, &lwork, &info);
  MPI_Barrier(MPI_COMM_WORLD);
  lwork = (int)work[0];
  work = new double[lwork];

  pdsyev_(&jobz, &uplo, &n, A, &ione, &ione, describeA, W, Z, &ione, &ione,
          describeZ, work, &lwork, &info);

  MPI_Barrier(MPI_COMM_WORLD);

  double EV_recv[mysize][n * n];
  int EV_r[mysize];
  int EV_c[mysize];
  int EV_rs[mysize];
  int EV_cs[mysize];
  double EV[n][n];

  MPI_Allgather(&mpA, 1, MPI_INT, &EV_r, 1, MPI_INT, MPI_COMM_WORLD);
  MPI_Allgather(&nqA, 1, MPI_INT, &EV_c, 1, MPI_INT, MPI_COMM_WORLD);
  MPI_Allgather(&ia, 1, MPI_INT, &EV_rs, 1, MPI_INT, MPI_COMM_WORLD);
  MPI_Allgather(&ja, 1, MPI_INT, &EV_cs, 1, MPI_INT, MPI_COMM_WORLD);

  if (myrank)
    MPI_Send(Z, mpA * nqA, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD);

  if (!myrank) {
    for (int i = 0; i < mpA * nqA; i++) {
      EV_recv[0][i] = Z[i];
    }
    MPI_Status status;
    for (int i = 1; i < mysize; i++) {
      MPI_Recv(EV_recv[i], EV_c[i] * EV_r[i], MPI_DOUBLE, i, 1, MPI_COMM_WORLD,
               &status);
    }
  }
  if (!myrank) {
    for (int i = 0; i < mysize; i++) {
      for (int j = 0; j < EV_c[i]; j++) {
        for (int k = 0; k < EV_r[i]; k++) {
          EV[EV_rs[i] + k][EV_cs[i] + j] = EV_recv[i][j * EV_r[i] + k];
        }
      }
    }
  }
  if (!myrank) {
    ofstream out;
    out.open("./result/diago_scalapack.txt");
    out << "-------------------eigenvalue:-------------------" << endl;
    for (int i = 0; i < n; i++) {
      out << W[i] << endl;
    }
    out << "-------------------eigenvector:-------------------" << endl;
    for (int i = 0; i < n; i++) {
      for (int j = 0; j < n; j++) {
        out << setw(15) << EV[i][j];
      }
      out << endl;
    }
    out.close();
    cout << "The result is printed in file \"./result/diago_scalapack.txt\"  . "
            "Check it out!"
         << endl;
  }
  blacs_gridexit_(&icontxt);
  MPI_Finalize();

  return 0;
}