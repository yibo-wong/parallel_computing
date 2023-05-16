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

class Input {
public:
  string matrix_type = "";
  int nrows = 0, ncols = 0;
  double *element;
  float *element_float;
  bool print_mpi_log = 1;

  Input() {}
  ~Input() {}
  int read_in(ifstream &file) {
    if (!file.is_open()) {
      cout << "fail to open!" << endl;
      return 1;
    }
    string line = "";
    while (getline(file, line)) {
      // erase the annotation
      int pos = line.find('#');
      if (pos != -1) {
        line.erase(pos, line.size());
      }
      if (line.empty() || all_of(line.begin(), line.end(), ::isspace)) {
        continue;
      }
      istringstream line_is(line);
      string name = "", value = "";
      line_is >> name;
      if (name == "value:") {
        if (matrix_type == "double") {
          element = new double[nrows * ncols];
          string mat_line;
          for (int i = 0; i < nrows; i++) {
            getline(file, mat_line);
            istringstream mat_is(mat_line);
            double temp = 0;
            char comma = 0;
            for (int j = 0; j < ncols - 1; j++) {
              mat_is >> temp >> comma;
              element[i * ncols + j] = temp;
            }
            mat_is >> temp;
            element[i * ncols + ncols - 1] = temp;
          }
        }
      } else if (matrix_type == "float") {
        element_float = new float[nrows * ncols];
        string mat_line;
        for (int i = 0; i < nrows; i++) {
          getline(file, mat_line);
          istringstream mat_is(mat_line);
          double temp = 0;
          char comma = 0;
          for (int j = 0; j < ncols - 1; j++) {
            mat_is >> temp >> comma;
            element_float[i * ncols + j] = temp;
          }
          mat_is >> temp;
          element_float[i * ncols + ncols - 1] = temp;
        }
      } else if (name == "type:") {
        line_is >> value;
        matrix_type = value;
      } else if (name == "nrows:") {
        line_is >> value;
        nrows = atoi(value.c_str());
      } else if (name == "ncols:") {
        line_is >> value;
        ncols = atoi(value.c_str());
      } else {
        cout << "wrong name!" << endl;
        return 1;
      }
    }
    return 0;
  }
  int Row() { return nrows; }
  int Col() { return ncols; }
};

class Input_demand {
public:
  string calculation = "";
  string matrix_type = "";
  string matrix_1_name = "";
  string matrix_2_name = "";
  string output_to_file = "";
  bool result_print = 0;
  bool timer_print = 0;
  double alpha, beta;

  Input_demand() {}
  ~Input_demand() {}
  int read_in(ifstream &file) {
    if (!file.is_open()) {
      cout << "fail to open!" << endl;
      return 1;
    }
    string line = "";
    while (getline(file, line)) {
      // erase the annotation
      int pos = line.find('#');
      if (pos != -1) {
        line.erase(pos, line.size());
      }
      if (line.empty() || all_of(line.begin(), line.end(), ::isspace)) {
        continue;
      }
      istringstream line_is(line);
      string name = "", value = "";
      line_is >> name >> value;
      if (name == "calculation") {
        calculation = value;
      } else if (name == "matrix_type") {
        matrix_type = value;
      } else if (name == "matrix_1") {
        // nrows = atoi(value.c_str());
        matrix_1_name = value;
      } else if (name == "matrix_2") {
        matrix_2_name = value;
      } else if (name == "print_mpi_log") {
        result_print = bool(atoi(value.c_str()));
      } else if (name == "timer_print") {
        timer_print = bool(atoi(value.c_str()));
      } else if (name == "output_to_file") {
        output_to_file = value;
      } else if (name == "alpha") {
        stringstream ss(value);
        ss >> alpha;
      } else if (name == "beta") {
        stringstream ss(value);
        ss >> beta;
      } else {
        cout << "wrong name!" << endl;
        return 1;
      }
    }
    return 0;
  }
};