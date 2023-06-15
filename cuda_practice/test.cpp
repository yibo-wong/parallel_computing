#include <iomanip>
#include <iostream>
#include <omp.h>

using namespace std;

int main() {
  int a = 32;

#pragma omp parallel for reduction(+ : a)
  for (int i = 0; i < 4; i++) {
#pragma omp critical
    {
      cout << i << endl;
      i += 1;
      a += i;
    }
  }
  cout << a << endl;
  return 0;
}