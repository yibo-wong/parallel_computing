#include <iostream>
#include <omp.h>
#include <sstream>
using namespace std;

int main() {
  omp_set_num_threads(8);
#pragma omp parallel
  {
    const int id = omp_get_thread_num();
    stringstream ss1, ss2;
    ss1 << "\033[3" << id << "m";
    ss2 << "\033[0m";
    cout << ss1.str() << "id=" << id << " hello world" << endl;
  }
  return 0;
}