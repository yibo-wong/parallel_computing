#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
using namespace std;

int main() {
  ofstream output;
  output.open("test.txt");
  output << "nrows: 79" << endl;
  output << "ncols: 30" << endl;
  output << "type: double" << endl;
  output << "value:" << endl;
  for (int i = 0; i < 79; i++) {
    for (int j = 0; j < 30 - 1; j++) {
      output << i << "." << j << ", ";
    }
    output << i << ".29";
    output << endl;
  }
  output.close();
  cout << endl << "end" << endl;
}