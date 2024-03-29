// Yibo Wang, 2100011025, coe_pku, parallel_programming.
#include "./include/input.h"
#include "./include/itpln.h"
#include "./include/omp_timer.h"
#include "./include/preprocess.h"
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <omp.h>
#include <sstream>
#include <string>

using namespace std;

int main() {
  timer::tick("read", "file");

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
  timer::tick("read", "file");

  double cutoff = input_f->cutoff;

  /////// using preprocess to optimize ///////
  cout << "PREPROCESSING..." << endl;
  timer::tick("pre-", "process");
  int x_pre = 30;
  int y_pre = 30;
  int z_pre = 30;
  Prep prep = Prep(x_pre, y_pre, z_pre, lx, ly, lz, cutoff);
  for (int i = 0; i < point_num; i++) {
    prep.update(input_p->px[i], input_p->py[i], input_p->pz[i], i);
  }
  timer::tick("pre-", "process");
  ////////////////////////////////////////////

  timer::tick("calc", "calc");

  cout << "start calculation..." << endl;

  Spline spline(input_f->mesh, input_f->cutoff, input_f->f);

  double hamilton[52][52];
  memset(hamilton, 0, sizeof(hamilton));
  double dx = (lx) / (input_v->nx);
  double dy = (ly) / (input_v->ny);
  double dz = (lz) / (input_v->nz);

  double big_dx = lx / x_pre;
  double big_dy = ly / y_pre;
  double big_dz = lz / z_pre;

#pragma omp parallel for collapse(3)
  for (int i = 0; i < x_pre; i++) {
    for (int j = 0; j < y_pre; j++) {
      for (int k = 0; k < z_pre; k++) {
        if (i == 0 && j == 0 && k == 0)
          cout << "total threads: " << omp_get_num_threads() << endl;
        long long int info = prep.value(i, j, k);

        int x_start = (i * big_dx) / dx;
        int x_end = i + 1 == x_pre ? nx : ((i + 1) * big_dx) / dx;
        int y_start = (j * big_dy) / dy;
        int y_end = j + 1 == y_pre ? ny : ((j + 1) * big_dy) / dy;
        int z_start = (k * big_dz) / dz;
        int z_end = k + 1 == z_pre ? nz : ((k + 1) * big_dz) / dz;

        for (int p_i = 0; p_i < point_num; p_i++) {
          // some prune
          if (!((info >> p_i) & 1))
            continue;
          for (int p_j = p_i; p_j < point_num; p_j++) {
            // some prune
            if (!((info >> p_j) & 1))
              continue;
            double sum = 0;
            for (int m_x = x_start; m_x < x_end; m_x++) {
              for (int m_y = y_start; m_y < y_end; m_y++) {
                for (int m_z = z_start; m_z < z_end; m_z++) {

                  double result = 0;

                  double x = dx * m_x;
                  double y = dy * m_y;
                  double z = dz * m_z;

                  int num_on_edge = int(m_x == 0 || m_x == nx - 1) +
                                    int(m_y == 0 || m_y == ny - 1) +
                                    int(m_z == 0 || m_z == nz - 1);

                  // double coe = 1.0; // coefficient of integral.
                  // if (num_on_edge == 1)
                  //   coe = 0.5;
                  // else if (num_on_edge == 2)
                  //   coe = 0.25;
                  // else if (num_on_edge == 3)
                  //   coe = 0.125;

                  double point_x_1 = input_p->px[p_i];
                  double point_y_1 = input_p->py[p_i];
                  double point_z_1 = input_p->pz[p_i];
                  double point_x_2 = 0;
                  double point_y_2 = 0;
                  double point_z_2 = 0;
                  double r1 = sqrt((x - point_x_1) * (x - point_x_1) +
                                   (y - point_y_1) * (y - point_y_1) +
                                   (z - point_z_1) * (z - point_z_1));
                  double r2 = 0;
                  if (p_i != p_j) {
                    point_x_2 = input_p->px[p_j];
                    point_y_2 = input_p->py[p_j];
                    point_z_2 = input_p->pz[p_j];
                    r2 = sqrt((x - point_x_2) * (x - point_x_2) +
                              (y - point_y_2) * (y - point_y_2) +
                              (z - point_z_2) * (z - point_z_2));
                  }

                  if (p_i != p_j) {
                    if (r1 > cutoff || r2 > cutoff)
                      result = 0;
                    else
                      result = spline.spline(r1) * spline.spline(r2) *
                               input_v->value(m_x, m_y, m_z);
                  } else {
                    if (r1 > cutoff)
                      result = 0;
                    else {
                      double spl = spline.spline(r1);
                      result = spl * spl * input_v->value(m_x, m_y, m_z);
                    }
                  }
                  result *= dx * dy * dz;
                  sum += result;
                }
              }
            }
#pragma omp critical
            {
              if (p_i == p_j)
                hamilton[p_i][p_j] += sum;
              else {
                hamilton[p_i][p_j] += sum;
                hamilton[p_j][p_i] += sum;
              }
            }
          }
        }
      }
    }
  }
  timer::tick("calc", "calc");
  timer::tick("write", "file");
  cout << "writing file..." << endl;

  ofstream out;
  out.open("./result/hamilton.txt");
  out << point_num << endl;
  for (int i = 0; i < point_num; i++) {
    for (int j = 0; j < point_num; j++) {
      out << setw(15) << hamilton[i][j];
    }
    out << endl;
  }
  out.close();
  timer::tick("write", "file");
  timer::print();
  return 0;
}
