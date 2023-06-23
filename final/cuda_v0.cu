// Yibo Wang, 2100011025, coe_pku, parallel_programming.
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>

#include "./include/input.h"
#include "./include/itpln.h"

using namespace std;

__device__ double calc_spline(double y[], double value[], double &len, int &n,
                              double &x) {
  int p;
  p = int(x * n / len) + 1;
  double ans = 0;
  double h = len / n;
  double dx_1 = x - (len * (p - 1)) / n;
  double dx_2 = x - (len * p) / n;
  double frac_1 = dx_1 / h;
  double frac_2 = dx_2 / h;
  ans = ((1 + 2 * frac_1) * value[p - 1] + dx_1 * y[p - 1]) * frac_2 * frac_2 +
        ((1 - 2 * frac_2) * value[p] + dx_2 * y[p]) * frac_1 * frac_1;
  return ans;
}

__global__ void calc(double lx, double ly, double lz, int nx, int ny, int nz,
                     double *V, double *px, double *py, double *pz,
                     double *spline_value, double *spline_y, double cutoff,
                     int n_mesh, int *calc_point_1, int *calc_point_2,
                     int cube_r) {
  if (threadIdx.x % 500 == 0) {
    double dx = lx / nx;
    int x = blockIdx.x;
    int p1 = calc_point_1[x];
    int p2 = calc_point_2[x];
    int cx = int(px[p1] / dx);
    int cy = int(py[p1] / dx);
    int cz = int(pz[p1] / dx);
    double v_t = V[cx * ny * nz + cy * nz + cz];
    printf(
        "I'm gpu %d - %d . At least it works.\n  My point 1 is: %d : "
        "(%f,%f,%f). My point 2 is: %d : (%f,%f,%f).\n  My central point "
        "is:(%d,%d,%d), cube radius = %d, V at center = %f \n",
        blockIdx.x, threadIdx.x, calc_point_1[x], px[p1], py[p1], pz[p1],
        calc_point_2[x], px[p2], py[p2], pz[p2], cx, cy, cz, cube_r, v_t);
  }
}

int main() {
  ifstream file;
  Input_V *input_v = new Input_V;
  Input_demand *input_d = new Input_demand;
  Input_points *input_p = new Input_points;
  Input_f *input_f = new Input_f;

  cout << "reading INPUT file..." << endl;
  file.open("./input/INPUT_test.txt");
  input_d->read_in(file);
  file.close();
  double lx = input_d->lx;
  double ly = input_d->ly;
  double lz = input_d->lz;
  string v_path = input_d->venergy_path;
  string p_path = input_d->points_path;
  string f_path = input_d->distribution_path;

  cout << "reading V file..." << endl;
  file.open(v_path.c_str());
  input_v->read_in(file);
  file.close();
  int nx = input_v->nx;
  int ny = input_v->ny;
  int nz = input_v->nz;

  cout << "reading POINTS file..." << endl;
  file.open(p_path.c_str());
  input_p->read_in(file);
  file.close();
  int point_num = input_p->num;

  cout << "reading DISTRIBUTION file..." << endl;
  file.open(f_path.c_str());
  input_f->read_in(file);
  file.close();

  double cutoff = input_f->cutoff;
  int mesh = input_f->mesh;

  bool cross[50][50];
  for (int i = 0; i < point_num; i++) {
    cross[i][i] = 1;
    for (int j = i + 1; j < 50; j++) {
      double x1 = input_p->px[i];
      double y1 = input_p->py[i];
      double z1 = input_p->pz[i];
      double x2 = input_p->px[j];
      double y2 = input_p->py[j];
      double z2 = input_p->pz[j];
      double dist2 =
          (x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2) + (z1 - z2) * (z1 - z2);
      if (dist2 < 4 * cutoff * cutoff) {
        cross[i][j] = cross[j][i] = 1;
      } else
        cross[i][j] = cross[j][i] = 0;
    }
  }

  int grid_size = 0;
  // these variables go into GPU.
  int calc_point_1[200];
  int calc_point_2[200];
  for (int i = 0; i < 50; i++) {
    for (int j = i; j < 50; j++) {
      if (cross[i][j]) {
        calc_point_1[grid_size] = i;
        calc_point_2[grid_size] = j;
        grid_size++;
      }
    }
  }

  // these go into GPU.
  Spline spline(input_f->mesh, input_f->cutoff, input_f->f);

  int spline_n = spline.n;
  double spline_len = spline.len;

  int block_size = 1024;

  // replicate these variables

  double *dev_v;
  cudaMalloc((double **)&dev_v, (nx * ny * nz + 2) * (sizeof(double)));
  cudaMemcpy(dev_v, input_v->V, (nx * ny * nz + 2) * (sizeof(double)),
             cudaMemcpyHostToDevice);

  double *dev_px;
  cudaMalloc((double **)&dev_px, point_num * sizeof(double));
  cudaMemcpy(dev_px, input_p->px, point_num * sizeof(double),
             cudaMemcpyHostToDevice);

  double *dev_py;
  cudaMalloc((double **)&dev_py, point_num * sizeof(double));
  cudaMemcpy(dev_py, input_p->py, point_num * sizeof(double),
             cudaMemcpyHostToDevice);

  double *dev_pz;
  cudaMalloc((double **)&dev_pz, point_num * sizeof(double));
  cudaMemcpy(dev_pz, input_p->pz, point_num * sizeof(double),
             cudaMemcpyHostToDevice);

  double *dev_spline_v;
  cudaMalloc((double **)&dev_spline_v, (input_f->mesh + 2) * sizeof(double));
  cudaMemcpy(dev_spline_v, spline.value, 1000 * sizeof(double),
             cudaMemcpyHostToDevice);

  double *dev_spline_y;
  cudaMalloc((double **)&dev_spline_y, (input_f->mesh + 2) * sizeof(double));
  cudaMemcpy(dev_spline_y, spline.y, 1000 * sizeof(double),
             cudaMemcpyHostToDevice);

  int *dev_point_1;
  cudaMalloc((int **)&dev_point_1, grid_size * sizeof(int));
  cudaMemcpy(dev_point_1, calc_point_1, grid_size * sizeof(int),
             cudaMemcpyHostToDevice);

  int *dev_point_2;
  cudaMalloc((int **)&dev_point_2, grid_size * sizeof(int));
  cudaMemcpy(dev_point_2, calc_point_2, grid_size * sizeof(int),
             cudaMemcpyHostToDevice);

  int cube_radius = int(cutoff / (lx / nx)) + 2;

  dim3 gsize(grid_size, 1, 1);
  dim3 bsize(block_size, 1, 1);

  calc<<<grid_size, block_size, 10 * sizeof(float)>>>(
      lx, ly, lz, nx, ny, nz, dev_v, dev_px, dev_py, dev_pz, dev_spline_v,
      dev_spline_y, cutoff, mesh, dev_point_1, dev_point_2, cube_radius);
  cudaDeviceSynchronize();
  cudaError_t error = cudaGetLastError();
  printf("CUDA error: %s\n", cudaGetErrorString(error));

  return 0;
}
