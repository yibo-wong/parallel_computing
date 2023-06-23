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

#define THREADS_NUM_THRESHOLD 600

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
  // printf("x:%f,value:%f,ans:%f\n", x, value[p - 1], ans);
  return ans;
}

__global__ void calc(double lx, double ly, double lz, int nx, int ny, int nz,
                     double *V, double *px, double *py, double *pz,
                     double *spline_value, double *spline_y, double cutoff,
                     int n_mesh, int *calc_point_1, int *calc_point_2,
                     int cube_r, int block_case, double *result) {
  extern __shared__ float shared_num[];
  double dx = lx / nx;
  int my_point = blockIdx.x;
  int index = threadIdx.x;
  int p1 = calc_point_1[my_point];
  int p2 = calc_point_2[my_point];
  double px1 = px[p1];
  double py1 = py[p1];
  double pz1 = pz[p1];
  double px2 = px[p2];
  double py2 = py[p2];
  double pz2 = pz[p2];
  int cx = int(px[p1] / dx);
  int cy = int(py[p1] / dx);
  int cz = int(pz[p1] / dx);
  int cube_len = cube_r * 2 + 1;
  double sum = 0.0;
  if (block_case == 1) {
    int x_int = index / (cube_len * cube_len) + cx - cube_r;
    int y_int = (index / (cube_len)) % cube_len + cy - cube_r;
    int z_int = index % cube_len + cz - cube_r;
    if (x_int < 0 || x_int >= nx || y_int < 0 || y_int >= ny || z_int < 0 ||
        z_int >= nz) {
      sum = 0.0;
    } else {
      double x = x_int * dx;
      double y = y_int * dx;
      double z = z_int * dx;
      double dist1 = sqrt((px1 - x) * (px1 - x) + (py1 - y) * (py1 - y) +
                          (pz1 - z) * (pz1 - z));
      double dist2 = 0;
      if (p1 == p2) {
        if (dist1 > cutoff)
          sum = 0.0;
        else {
          double f = calc_spline(spline_y, spline_value, cutoff, n_mesh, dist1);
          sum = f * f * dx * dx * dx * V[x_int * ny * nz + y_int * nz + z_int];
        }
      } else {
        dist2 = sqrt((px2 - x) * (px2 - x) + (py2 - y) * (py2 - y) +
                     (pz2 - z) * (pz2 - z));
        if (dist1 > cutoff || dist2 > cutoff)
          sum = 0.0;
        else {
          double f1 =
              calc_spline(spline_y, spline_value, cutoff, n_mesh, dist1);
          double f2 =
              calc_spline(spline_y, spline_value, cutoff, n_mesh, dist2);
          sum =
              f1 * f2 * dx * dx * dx * V[x_int * ny * nz + y_int * nz + z_int];
        }
      }
    }
    __syncthreads();
    atomicAdd(&shared_num[0], float(sum));
    __syncthreads();
    if (!threadIdx.x) result[blockIdx.x] = double(shared_num[0]);
    return;
  } else if (block_case == 2) {
    int x_int = index / (cube_len) + cx - cube_r;
    int y_int = index % cube_len + cy - cube_r;
    int z_int = 0;
    for (int i = 0; i < cube_len; i++) {
      z_int = i + cz - cube_r;
      if (x_int < 0 || x_int >= nx || y_int < 0 || y_int >= ny || z_int < 0 ||
          z_int >= nz) {
        sum += 0.0;
      } else {
        double x = x_int * dx;
        double y = y_int * dx;
        double z = z_int * dx;
        double dist1 = sqrt((px1 - x) * (px1 - x) + (py1 - y) * (py1 - y) +
                            (pz1 - z) * (pz1 - z));
        // printf("dist: %f \n", dist1);
        double dist2 = 0;
        if (p1 == p2) {
          if (dist1 > cutoff)
            sum += 0.0;
          else {
            double f =
                calc_spline(spline_y, spline_value, cutoff, n_mesh, dist1);
            // printf("f: %f\n", f);
            sum +=
                f * f * dx * dx * dx * V[x_int * ny * nz + y_int * nz + z_int];
          }
        } else {
          dist2 = sqrt((px2 - x) * (px2 - x) + (py2 - y) * (py2 - y) +
                       (pz2 - z) * (pz2 - z));
          if (dist1 > cutoff || dist2 > cutoff)
            sum += 0.0;
          else {
            double f1 =
                calc_spline(spline_y, spline_value, cutoff, n_mesh, dist1);
            double f2 =
                calc_spline(spline_y, spline_value, cutoff, n_mesh, dist2);
            sum += f1 * f2 * dx * dx * dx *
                   V[x_int * ny * nz + y_int * nz + z_int];
          }
        }
      }
    }
    __syncthreads();
    atomicAdd(&shared_num[0], float(sum));
    __syncthreads();
    if (!threadIdx.x) {
      result[blockIdx.x] = double(shared_num[0]);
      // printf("On block %d, result = %f \n", blockIdx.x, result[blockIdx.x]);
    }
    return;
  } else if (block_case == 3) {
    int x_int = 0;
    int y_int = 0;
    int z_int = 0;
    int total_points = cube_len * cube_len;
    int block_size = THREADS_NUM_THRESHOLD;
    int my_thd = threadIdx.x;

    int ave = total_points / block_size;
    int index_num = (my_thd < total_points % block_size) ? ave + 1 : ave;
    int index_start = (my_thd < total_points % block_size)
                          ? my_thd * (ave + 1)
                          : (total_points % block_size) * (ave + 1) +
                                (my_thd - total_points % block_size) * ave;
    for (int j = index_start; j < index_num + index_start; j++) {
      x_int = j / (cube_len) + cx - cube_r;
      y_int = j % cube_len + cy - cube_r;
      for (int i = 0; i < cube_len; i++) {
        z_int = i + cz - cube_r;
        if (x_int < 0 || x_int >= nx || y_int < 0 || y_int >= ny || z_int < 0 ||
            z_int >= nz) {
          sum += 0.0;
        } else {
          double x = x_int * dx;
          double y = y_int * dx;
          double z = z_int * dx;
          double dist1 = sqrt((px1 - x) * (px1 - x) + (py1 - y) * (py1 - y) +
                              (pz1 - z) * (pz1 - z));
          // printf("dist: %f \n", dist1);
          double dist2 = 0;
          if (p1 == p2) {
            if (dist1 > cutoff)
              sum += 0.0;
            else {
              double f =
                  calc_spline(spline_y, spline_value, cutoff, n_mesh, dist1);
              // printf("f: %f\n", f);
              sum += f * f * dx * dx * dx *
                     V[x_int * ny * nz + y_int * nz + z_int];
            }
          } else {
            dist2 = sqrt((px2 - x) * (px2 - x) + (py2 - y) * (py2 - y) +
                         (pz2 - z) * (pz2 - z));
            if (dist1 > cutoff || dist2 > cutoff)
              sum += 0.0;
            else {
              double f1 =
                  calc_spline(spline_y, spline_value, cutoff, n_mesh, dist1);
              double f2 =
                  calc_spline(spline_y, spline_value, cutoff, n_mesh, dist2);
              sum += f1 * f2 * dx * dx * dx *
                     V[x_int * ny * nz + y_int * nz + z_int];
            }
          }
        }
      }
    }
    __syncthreads();
    atomicAdd(&shared_num[0], float(sum));
    __syncthreads();
    if (!threadIdx.x) {
      result[blockIdx.x] = double(shared_num[0]);
      // printf("On block %d, result = %f \n", blockIdx.x, result[blockIdx.x]);
    }
    return;
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

  // cout << "grid:" << grid_size << endl;

  // these go into GPU.
  Spline spline(input_f->mesh, input_f->cutoff, input_f->f);

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
  cudaMalloc((double **)&dev_spline_v, 1000 * sizeof(double));
  cudaMemcpy(dev_spline_v, spline.value, 1000 * sizeof(double),
             cudaMemcpyHostToDevice);

  double *dev_spline_y;
  cudaMalloc((double **)&dev_spline_y, 1000 * sizeof(double));
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

  int cube_radius = int(cutoff / (lx / nx)) + 1;
  int cube_len = 2 * cube_radius + 1;
  int cube_2 = cube_len * cube_len;
  int cube_3 = cube_len * cube_len * cube_len;

  int block_size = THREADS_NUM_THRESHOLD;
  int block_case = 0;
  if (cube_3 <= THREADS_NUM_THRESHOLD) {
    block_size = cube_3;
    block_case = 1;
  } else if (cube_3 > THREADS_NUM_THRESHOLD && cube_2 < THREADS_NUM_THRESHOLD) {
    block_size = cube_2;
    block_case = 2;
  } else {
    block_case = 3;
    block_size = THREADS_NUM_THRESHOLD;
  }

  double *dev_result;
  cudaMalloc((double **)&dev_result, grid_size * sizeof(double));
  cout << "calculating..." << endl;
  // timer
  cudaEvent_t start, stop;
  float elapsedTime = 0.0;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, 0);

  calc<<<grid_size, block_size, 1 * sizeof(float)>>>(
      lx, ly, lz, nx, ny, nz, dev_v, dev_px, dev_py, dev_pz, dev_spline_v,
      dev_spline_y, cutoff, mesh, dev_point_1, dev_point_2, cube_radius,
      block_case, dev_result);
  cudaDeviceSynchronize();

  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&elapsedTime, start, stop);
  cout << "calc time: " << elapsedTime << " ms" << endl;

  cudaError_t error = cudaGetLastError();
  printf("CUDA error: %s\n", cudaGetErrorString(error));

  double host_result[200];
  cudaMemcpy(host_result, dev_result, grid_size * sizeof(double),
             cudaMemcpyDeviceToHost);
  double hamilton[50][50];
  memset(hamilton, 0, sizeof(hamilton));
  for (int i = 0; i < grid_size; i++) {
    int pi = calc_point_1[i];
    int pj = calc_point_2[i];
    hamilton[pi][pj] = hamilton[pj][pi] = host_result[i];
  }

  ofstream out;
  out.open("./result/hamilton_cuda.txt");
  out << point_num << endl;
  for (int i = 0; i < point_num; i++) {
    for (int j = 0; j < point_num; j++) {
      out << setw(15) << hamilton[i][j];
    }
    out << endl;
  }
  out.close();

  return 0;
}
