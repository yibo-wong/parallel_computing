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

class Prep {
public:
  long long int *info;
  int nx, ny, nz;
  double lx, ly, lz;
  double cutoff;
  Prep(int x, int y, int z, double lx, double ly, double lz, double c)
      : nx(x), ny(y), nz(z), lx(lx), ly(ly), lz(lz), cutoff(c) {
    info = new long long int[x * y * z + 10];
    memset(info, 0, sizeof(info));
  }

  long long int value(int x, int y, int z) // start from 0
  {
    return info[x * ny * nz + y * nz + z];
  }

  bool calc(double x, double y, double z, int id) {
    double dx = lx / nx;
    double dy = ly / ny;
    double dz = lz / nz;
    int tx, ty, tz; // index of info[]
    for (int i = 0; i < nx; i++) {
      if (i * dx <= x && (i + 1) * dx >= x) {
        tx = i;
        break;
      }
    }
    for (int i = 0; i < ny; i++) {
      if (i * dy <= y && (i + 1) * dy >= y) {
        ty = i;
        break;
      }
    }
    for (int i = 0; i < nz; i++) {
      if (i * dz <= z && (i + 1) * dz >= z) {
        tz = i;
        break;
      }
    }
    bool flag = (info[tx * ny * nz + ty * nz + tz] >> id) & (1);
    return flag;
  }

  void update(double px, double py, double pz, int id) {
    double dx = lx / nx;
    double dy = ly / ny;
    double dz = lz / nz;
    double cx = 0, cy = 0, cz = 0, d = 0;
    double eps = sqrt(dx * dx + dy * dy + dz * dz) / 2.0 + 0.01;
    ofstream out;
    // out.open("./result/preprocess.log");
    for (int i = 0; i < nx; i++) {
      for (int j = 0; j < ny; j++) {
        for (int k = 0; k < nz; k++) {
          cx = (i + 0.5) * dx;
          cy = (j + 0.5) * dy;
          cz = (k + 0.5) * dz;
          if ((px - cx) * (px - cx) + (py - cy) * (py - cy) +
                  (pz - cz) * (pz - cz) <
              cutoff * cutoff + 2 * cutoff * eps + eps * eps) {
            // out << dec << cx << "  " << cy << "  " << cz << "  " << id <<
            // endl;
            long long int one = 1;
            info[i * ny * nz + j * nz + k] =
                (info[i * ny * nz + j * nz + k]) | (one << id);
            // out << info[i * ny * nz + j * nz + k] << endl << endl;
          }
        }
      }
    }
    out.close();
  }
};