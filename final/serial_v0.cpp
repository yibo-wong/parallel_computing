// Yibo Wang, 2100011025, coe_pku, parallel_programming.
#include "./include/input.h"
#include "./include/itpln.h"
#include "./include/preprocess.h"
#include "./include/timer.h"
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
  timer::tick("read", "file");
  ifstream file;
  Input_V *input_v = new Input_V;
  Input_demand *input_d = new Input_demand;
  Input_points *input_p = new Input_points;
  Input_f *input_f = new Input_f;

  file.open("./input/INPUT.txt");
  input_d->read_in(file);
  file.close();
  double lx = input_d->lx;
  double ly = input_d->ly;
  double lz = input_d->lz;
  string v_path = input_d->venergy_path;
  string p_path = input_d->points_path;
  string f_path = input_d->distribution_path;

  file.open(v_path);
  input_v->read_in(file);
  file.close();

  file.open(p_path);
  input_p->read_in(file);
  file.close();
  int point_num = input_p->num;

  file.open(f_path);
  input_f->read_in(file);
  file.close();
  timer::tick("read", "file");
  double cutoff = input_f->cutoff;

  timer::tick("calc", "calc");

  Spline spline(input_f->mesh, input_f->cutoff, input_f->f);

  double hamilton[52][52];
  double dx = (input_d->lx) / (input_v->nx - 1);
  double dy = (input_d->ly) / (input_v->ny - 1);
  double dz = (input_d->lz) / (input_v->nz - 1);

  for (int m_x = 0; m_x < input_v->nx; m_x++) {
    cout << m_x << endl;
    for (int m_y = 0; m_y < input_v->ny; m_y++) {
      for (int m_z = 0; m_z < input_v->nz; m_z++) {
        for (int p_i = 0; p_i < input_p->num; p_i++) {
          for (int p_j = p_i; p_j < input_p->num; p_j++) {
            double result = 0;
            /////// Simpson Algorithm or not??? ///////
            // int s_x = 0, s_y = 0, s_z = 0;
            // double simp = 0;
            // if (m_x == 0 || m_x == input_v->nx - 1)
            //   s_x = 1;
            // else
            //   s_x = m_x % 2 == 0 ? 2 : 4;

            // if (m_y == 0 || m_y == input_v->ny - 1)
            //   s_y = 1;
            // else
            //   s_y = m_y % 2 == 0 ? 2 : 4;

            // if (m_z == 0 || m_z == input_v->nz - 1)
            //   s_z = 1;
            // else
            //   s_z = m_z % 2 == 0 ? 2 : 4;

            // simp = double(s_x * s_y * s_z) / 27.0;

            double x = dx * m_x;
            double y = dy * m_y;
            double z = dz * m_z;

            double point_x_1 = input_p->px[p_i];
            double point_y_1 = input_p->py[p_i];
            double point_z_1 = input_p->pz[p_i];
            double point_x_2 = 0;
            double point_y_2 = 0;
            double point_z_2 = 0;
            double cutoff = input_f->cutoff;
            double r1 = dist(x, y, z, point_x_1, point_y_1, point_z_1);
            double r2 = 0;
            if (p_i != p_j) {
              point_x_2 = input_p->px[p_j];
              point_y_2 = input_p->py[p_j];
              point_z_2 = input_p->pz[p_j];
              r2 = dist(x, y, z, point_x_2, point_y_2, point_z_2);
            }
            bool calc = 0;

            if (p_i != p_j)
              calc = (r1 < cutoff) && (r2 < cutoff);
            else
              calc = (r1 < cutoff);
            if (calc) {
              if (p_i != p_j) {
                result = spline.spline(r1) * spline.spline(r2) *
                         input_v->value(m_x, m_y, m_z);
              } else {
                result =
                    pow(spline.spline(r1), 2) * input_v->value(m_x, m_y, m_z);
              }

              // result *= dx * dy * dz * simp;
              result *= dx * dy * dz;
              if (p_i == p_j)
                hamilton[p_i][p_j] += result;
              else {
                hamilton[p_i][p_j] += result;
                hamilton[p_j][p_i] += result;
              }
            }
          }
        }
      }
    }
  }
  timer::tick("calc", "calc");

  timer::tick("write", "file");
  ofstream out;
  out.open("./result/hamilton_v0.txt");
  for (int i = 0; i < input_p->num; i++) {
    for (int j = 0; j < input_p->num; j++) {
      out << setw(15) << hamilton[i][j];
    }
    out << endl;
  }
  out.close();
  timer::tick("write", "file");
  timer::print();
  return 0;
}
