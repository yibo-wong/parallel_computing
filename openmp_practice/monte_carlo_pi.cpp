#include <iostream>
#include <omp.h>

using namespace std;

int main() {
  omp_set_num_threads(8);
  int N = 1000;
  int M = 4;
  int P = 5000000;
  double start = 0;
  double end = 0;

  start = omp_get_wtime();
  double sum = 0;
  for (int i = 0; i < M; i++) {
    cout << i << endl;
    double s = 0;
    for (int j = 0; j < P; j++) {
      if (j % 1000000 == 0)
        cout << j << endl;
      double x = double(rand() % N) / N;
      double y = double(rand() % N) / N;
      if (x * x + y * y <= 1) {
        s += 1;
      }
    }
    cout << "end" << endl;
    sum += s;
  }
  sum /= M * P;
  cout << sum * 4 << endl;
  end = omp_get_wtime();
  double s_time = end - start;

  start = omp_get_wtime();
  sum = 0;
#pragma omp parallel for reduction(+ : sum) schedule(dynamic)
  for (int i = 0; i < M; i++) {
    cout << i << endl;
    double s = 0;
    for (int j = 0; j < P; j++) {
      if (j % 1000000 == 0)
        cout << j << endl;
      double x = double(rand() % N) / N;
      double y = double(rand() % N) / N;
      if (x * x + y * y <= 1) {
        s += 1;
      }
    }
    cout << "end" << endl;
    sum += s;
  }
  sum /= M * P;
  cout << sum * 4 << endl;
  end = omp_get_wtime();
  double p_time = end - start;
  cout << "time" << endl
       << s_time << "  " << p_time << "  " << (s_time / p_time) << endl;
  return 0;
}