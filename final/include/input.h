// Yibo Wang, 2100011025, coe_pku, parallel_programming.
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#define maxn 100

using namespace std;

class Input_V {
public:
  int nx, ny, nz;
  double *V;

  Input_V() {}
  ~Input_V() {}
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
      if (name == "nx") {
        line_is >> value;
        nx = atoi(value.c_str());
      } else if (name == "ny") {
        line_is >> value;
        ny = atoi(value.c_str());
      } else if (name == "nz") {
        line_is >> value;
        nz = atoi(value.c_str());
      } else if (name == "V:") {
        V = new double[nx * ny * nz + 2];
        for (int i = 0; i < nx * ny * nz; i++) {
          file >> V[i];
        }
        return 0;
      }
    }
    return 0;
  }
  double value(const int &x, const int &y, const int &z) {
    return V[x * ny * nz + y * nz + z];
  }
};

class Input_demand {
public:
  bool isHexahedral = 0;
  double lx = 0;
  double ly = 0;
  double lz = 0;
  double thetaxy = 0;
  double thetayz = 0;
  double thetaxz = 0;
  bool support_SH = 0;
  string diago_lib = "";
  bool support_Periodic_Boundary = 0;
  bool multi_parallel_strategies = 0;
  string points_path = "";
  string venergy_path = "";
  string distribution_path = "";

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
      if (name == "isHexahedral") {
        isHexahedral = atoi(value.c_str());
      } else if (name == "lx") {
        stringstream ss(value);
        ss >> lx;
      } else if (name == "ly") {
        stringstream ss(value);
        ss >> ly;
      } else if (name == "lz") {
        stringstream ss(value);
        ss >> lz;
      } else if (name == "thetaxy") {
        stringstream ss(value);
        ss >> thetaxy;
      } else if (name == "thetayz") {
        stringstream ss(value);
        ss >> thetayz;
      } else if (name == "thetaxz") {
        stringstream ss(value);
        ss >> thetaxz;
      } else if (name == "support_SH") {
        support_SH = atoi(value.c_str());
      } else if (name == "diago_lib") {
        diago_lib = value;
      } else if (name == "support_Periodic_Boundary") {
        support_Periodic_Boundary = bool(atoi(value.c_str()));
      } else if (name == "multi_parallel_strategies") {
        multi_parallel_strategies = bool(atoi(value.c_str()));
      } else if (name == "points_path") {
        points_path = value;
      } else if (name == "venergy_path") {
        venergy_path = value;
      } else if (name == "distribution_path") {
        distribution_path = value;
      } else {
        cout << "wrong name!" << endl;
        return 1;
      }
    }
    return 0;
  }
};

class Input_points {
public:
  int num = 0;
  double px[60], py[60], pz[60];

  Input_points() {}
  ~Input_points() {}
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
      char useless;
      line_is >> useless >> px[num] >> useless >> py[num] >> useless >>
          pz[num] >> useless;
      num += 1;
    }
    return 0;
  }
};

class Input_f {
public:
  double cutoff = 0;
  double dr = 0;
  int mesh = 0;
  int l = 0;
  double *f;

  Input_f() {}
  ~Input_f() {}
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
      if (name == "mesh") {
        line_is >> mesh;
      } else if (name == "dr") {
        line_is >> dr;
      } else if (name == "cutoff") {
        line_is >> cutoff;
      } else if (name == "f:") {
        f = new double[mesh + 10];
        for (int i = 0; i < mesh; i++) {
          char useless;
          file >> f[i] >> useless;
        }
        return 0;
      }
    }
    return 0;
  }
};