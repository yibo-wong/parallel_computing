#include <iostream>
#include <omp.h>

using namespace std;
int main() {
  omp_set_num_threads(50);
#pragma omp parallel
  {
    cout << "hello "
         << "world" << endl;
  }
  return 0;
}