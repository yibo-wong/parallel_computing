#include <algorithm>
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
  int N = 100000;
  out.open("point.txt");
  for (int i = 1; i <= 50; i++) {
    out << "(" << 1000 * double(rand() % N) / N << ", "
        << 1000 * double(rand() % N) / N << ", "
        << 1000 * double(rand() % N) / N << ")" << endl;
  }
  out.close();
}