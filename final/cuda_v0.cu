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
#include "./include/preprocess.h"
#include "./include/timer.h"

using namespace std;

__global__ void calculation(double *cuda_v, double *cuda_f, double *cuda_y,
                            double cutoff, int mesh, double *cuda_ham,
                            double *cuda_px, double *cuda_py, double *cuda_pz,
                            long long int *cuda_info, double lx, double ly,
                            double lz, int nx, int ny, int nz, int x_pre,
                            int y_pre, int z_pre, int point_num) {
  bool not_calc = false;
  double result = 0;
  int i = blockIdx.x;
  int j = blockIdx.y;
  int k = blockIdx.z;
  int t_i = threadIdx.x;
  int t_j = threadIdx.y;
  if (t_j < t_i) not_calc = true;
  int p_i = 0;
  int p_j = 0;
  double dx = lx / (nx - 1);
  double dy = ly / (ny - 1);
  double dz = lz / (nz - 1);
  double big_dx = dx / x_pre;
  double big_dy = dy / y_pre;
  double big_dz = dz / z_pre;
  int big_x = i == nx ? x_pre : int(i * dx / big_dx);
  int big_y = j == ny ? y_pre : int(j * dy / big_dy);
  int big_z = k == nz ? z_pre : int(k * dz / big_dz);
  long long int point_info =
      cuda_info[big_x * y_pre * z_pre + big_y * z_pre + big_z];

  int sum_temp = 0;
  bool flag = 0;
  for (int i = 0; i < point_num; i++) {
    if ((point_info >> i) & 1) {
      if (sum_temp == p_i) {
        flag = 1;
        p_i = i;
        break;
      }
      sum_temp++;
    }
  }
  if (!flag) not_calc = true;

  sum_temp = 0;
  flag = 0;
  for (int i = 0; i < point_num; i++) {
    if ((point_info >> i) & 1) {
      if (sum_temp == p_j) {
        flag = 1;
        p_j = i;
        break;
      }
      sum_temp++;
    }
  }
  if (!flag) not_calc = true;

  double result = 0;

  double x = dx * i;
  double y = dy * j;
  double z = dz * k;

  int num_on_edge = int(i == 0 || i == nx - 1) + int(j == 0 || j == ny - 1) +
                    int(k == 0 || k == nz - 1);

  double coe = 1.0;  // coefficient of integral.
  if (num_on_edge == 1)
    coe = 0.5;
  else if (num_on_edge == 2)
    coe = 0.25;
  else if (num_on_edge == 3)
    coe = 0.125;

  double point_x_1 = cuda_px[p_i];
  double point_y_1 = cuda_py[p_i];
  double point_z_1 = cuda_pz[p_i];
  double point_x_2 = 0;
  double point_y_2 = 0;
  double point_z_2 = 0;
  double r1 = sqrt((x - point_x_1) * (x - point_x_1) +
                   (y - point_y_1) * (y - point_y_1) +
                   (z - point_z_1) * (z - point_z_1));
  double r2 = 0;
  if (p_i != p_j) {
    point_x_2 = cuda_px[p_j];
    point_y_2 = cuda_py[p_j];
    point_z_2 = cuda_pz[p_j];
    r2 = sqrt((x - point_x_2) * (x - point_x_2) +
              (y - point_y_2) * (y - point_y_2) +
              (z - point_z_2) * (z - point_z_2));
  }
  double f1 = 0;
  double f2 = 0;
  double v = cuda_v[i * ny * nz + j * nz + k];
  if (p_i != p_j) {
    if (r1 >= cutoff || r2 >= cutoff) not_calc = true;
  } else {
    if (r1 >= cutoff) not_calc = true;
  }
  ////////calc spline(r1)/////////
  if (!not_calc) {
    int p = int(r1 * mesh / cutoff) + 1;
    double ans = 0;
    double h = cutoff / mesh;
    double dx_1 = r1 - cutoff * (p - 1) / mesh;
    double dx_2 = r1 - cutoff * p / mesh;
    double frac_1 = dx_1 / h;
    double frac_2 = dx_2 / h;
    f1 = ((1 + 2 * frac_1) * cuda_f[p - 1] + dx_1 * cuda_f[p - 1]) * frac_2 *
             frac_2 +
         ((1 - 2 * frac_2) * cuda_f[p] + dx_2 * cuda_y[p]) * frac_1 * frac_1;
  }
  ////////////////////////////////

  ////////calc spline(r2)/////////
  if (p_i != p_j && (!not_calc)) {
    int p = int(r2 * mesh / cutoff) + 1;
    double ans = 0;
    double h = cutoff / mesh;
    double dx_1 = r2 - cutoff * (p - 1) / mesh;
    double dx_2 = r2 - cutoff * p / mesh;
    double frac_1 = dx_1 / h;
    double frac_2 = dx_2 / h;
    f1 = ((1 + 2 * frac_1) * cuda_f[p - 1] + dx_1 * cuda_f[p - 1]) * frac_2 *
             frac_2 +
         ((1 - 2 * frac_2) * cuda_f[p] + dx_2 * cuda_y[p]) * frac_1 * frac_1;
  }
  ////////////////////////////////

  if (!not_calc) {
    if (p_i == p_j) {
      result = dx * dy * dz * f1 * f1 * coe * v;
    } else {
      result = dx * dy * dz * f1 * f2 * coe * v;
    }
  }
  cuda_ham[(p_i * point_num + p_j) * (nx * ny * nz) + i * ny * nz + j * nz +
           k] = result;
  __syncthreads();

  ////////reduction////////

  return;
}

void cuda_solve(double *v, double *f, double *y, double cutoff, int mesh,
                double *px, double *py, double *pz, long long int *info,
                double lx, double ly, double lz, int nx, int ny, int nz,
                int x_pre, int y_pre, int z_pre, int point_num) {
  // calculation(double *cuda_v, double *cuda_f, double *cuda_y,
  //                             double cutoff, int mesh, double ***cuda_ham,
  //                             double *cuda_px, double *cuda_py, double
  //                             *cuda_pz, long long int *cuda_info, double lx,
  //                             double ly, double lz, int nx, int ny, int nz,
  //                             int x_pre, int y_pre, int z_pre)

  int size_v = (nx * ny * nz + 2) * sizeof(double);
  double *cuda_v;
  cudaMalloc((void **)&cuda_v, size_v);
  cudaMemcpy(cuda_v, v, size_v, cudaMemcpyHostToDevice);

  int size_f = (mesh + 10) * sizeof(double);
  double *cuda_f;
  cudaMalloc((void **)&cuda_f, size_f);
  cudaMemcpy(cuda_f, f, size_f, cudaMemcpyHostToDevice);

  int size_y = (mesh + 10) * sizeof(double);
  double *cuda_y;
  cudaMalloc((void **)&cuda_y, size_y);
  cudaMemcpy(cuda_y, y, size_y, cudaMemcpyHostToDevice);

  int size_p = (point_num + 5) * sizeof(double);
  double *cuda_px;
  cudaMalloc((void **)&cuda_px, size_p);
  cudaMemcpy(cuda_px, px, size_p, cudaMemcpyHostToDevice);
  double *cuda_py;
  cudaMalloc((void **)&cuda_py, size_p);
  cudaMemcpy(cuda_py, py, size_p, cudaMemcpyHostToDevice);
  double *cuda_pz;
  cudaMalloc((void **)&cuda_pz, size_p);
  cudaMemcpy(cuda_pz, pz, size_p, cudaMemcpyHostToDevice);

  int size_info = (point_num + 5) * sizeof(long long int);
  long long int *cuda_info;
  cudaMalloc((void **)&cuda_info, size_info);
  cudaMemcpy(cuda_info, info, size_info, cudaMemcpyHostToDevice);

  double *cuda_ham;
  int size_ham = point_num * point_num * nx * ny * nz + 10;

  dim3 grid_size(nx + 5, ny + 5, nz + 5);
  dim3 block_size(32, 32);
}

int main() {
  // timer::tick("read", "file");
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
  file.open(v_path);
  input_v->read_in(file);
  file.close();
  int nx = input_v->nx;
  int ny = input_v->ny;
  int nz = input_v->nz;

  cout << "reading POINTS file..." << endl;
  file.open(p_path);
  input_p->read_in(file);
  file.close();
  int point_num = input_p->num;

  cout << "reading DISTRIBUTION file..." << endl;
  file.open(f_path);
  input_f->read_in(file);
  file.close();
  // timer::tick("read", "file");
  double cutoff = input_f->cutoff;

  /////// using preprocess to optimize ///////
  cout << "preprocessing..." << endl;
  // timer::tick("pre-", "process");
  int preprocess_mesh = 30;
  int x_pre = preprocess_mesh;
  int y_pre = preprocess_mesh;
  int z_pre = preprocess_mesh;
  Prep prep = Prep(x_pre, y_pre, z_pre, lx, ly, lz, cutoff);
  for (int i = 0; i < point_num; i++) {
    prep.update(input_p->px[i], input_p->py[i], input_p->pz[i], i);
  }
  long long int *info;
  info = new long long int[x_pre * y_pre * z_pre + 8];
  memcpy(info, prep.info,
         int(sizeof(long long int)) * (x_pre * y_pre * z_pre + 5));
  // timer::tick("pre-", "process");
  ////////////////////////////////////////////

  // timer::tick("calc", "calc");
  cout << "start calculation..." << endl;

  Spline spline(input_f->mesh, input_f->cutoff, input_f->f);

  // timer::tick("calc", "calc");
  // // timer::tick("write", "file");
  // cout << "writing file..." << endl;
  // ofstream out;
  // out.open("./result/hamilton_cuda_v0.txt");
  // for (int i = 0; i < input_p->num; i++) {
  //   for (int j = 0; j < input_p->num; j++) {
  //     out << setw(15) << hamilton[i][j];
  //   }
  //   out << endl;
  // }
  // out.close();
  // // timer::tick("write", "file");
  // // timer::print();
  return 0;
}
