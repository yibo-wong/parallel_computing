#include <math.h>
#include <stdio.h>

int N = 50;
int M = 50;

double Jacobi(double *f_1d, double *u_old, double *u_new, double r1, double r2,
              double r3, double r) {
  int i, j;
  double error = 0;
  double resid = 0;
  for (j = 0; j < N; j++) {
    for (i = 0; i < M; i++) {
      u_old[j * M + i] = u_new[j * M + i];
    }
  }
  for (j = 0; j < N; j++) {
    for (i = 0; i < M; i++) {
      if (j == 0 || j == N - 1 || i == 0 || i == M - 1)
        continue;
      else {
        resid =
            f_1d[j * M + i] -
            (r1 * (u_old[(j - 1) * M + i - 1] + u_old[(j - 1) * M + i + 1]) +
             r3 * (u_old[(j)*M + i - 1] + u_old[(j)*M + i + 1]) +
             r1 * (u_old[(j + 1) * M + i - 1] + u_old[(j + 1) * M + i + 1]) +
             r2 * (u_old[(j - 1) * M + i] + u_old[(j + 1) * M + i]) +
             r * u_old[j * M + i]);
        u_new[j * M + i] = u_new[j * M + i] + resid / r;
        error += resid * resid;
      }
    }
  }
  return error;
}