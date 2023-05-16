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

int main() {
  ofstream out;
  int N = 300;
  out.open("V_test_big.txt");
  out << "nx " << N << endl;
  out << "ny " << N << endl;
  out << "nz " << N << endl;
  out << "V:" << endl;
  for (int i = 1; i <= N; i++) {
    for (int j = 1; j <= N; j++) {
      for (int k = 1; k <= N; k++) {
        double v = 10 * exp(double(i) / N) * sin(3.14 * double(j) / N) *
                   exp(-double(k) / N);
        out << v << " ";
      }
    }
  }
  out.close();
}