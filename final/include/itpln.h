#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
using namespace std;

double dist(double x1, double y1, double z1, double x2, double y2, double z2) {
  return sqrt((x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2) +
              (z1 - z2) * (z1 - z2));
}

class Linear {
 public:
  int n;
  double len;
  double value[1000];
  Linear(){};
  Linear(int m, double len, double *v) : n(m), len(len) {
    for (int i = 0; i <= n; i++) value[i] = v[i];
  }
  double linear(double x) {
    int p;
    for (int i = 0; i <= n; i++) {
      if (len * i / n >= x) {
        p = i;
        break;
      }
    }
    double a = x - len * (p - 1) / n;
    double b = len * p / n - x;
    return (a * value[p - 1] + b * value[p]) / (a + b);
  }
};
class Spline {
 public:
  int n;
  double len;
  double value[1000];
  Spline() {}
  // m for mesh, len for cutoff, v for value.
  Spline(int m, double len, double *v) : n(m), len(len) {
    for (int i = 0; i <= n; i++) value[i] = v[i];
    triLU();
  }
  double a[1000], b[1000], c[1000], y[1000];
  void triLU() {
    y[0] = (value[1] - value[0]) / (len / n);
    y[n] = 0;

    for (int i = 1; i < n; i++) {
      y[i] = 1.5 * n * (value[i + 1] - value[i - 1]) / len;
    }

    a[0] = 1;
    for (int i = 1; i < n; i++) a[i] = 2;
    a[n] = 1;
    c[0] = 0;
    for (int i = 1; i < n; i++) b[i] = c[i] = 0.5;
    b[n] = 0;
    for (int i = 1; i < n; i++) {
      a[i] -= b[i] * c[i - 1];
      b[i + 1] /= a[i];
    }

    for (int i = 1; i <= n; i++) y[i] -= y[i - 1] * b[i];
    y[n] /= a[n];
    for (int i = n - 1; i >= 0; i--) y[i] = (y[i] - c[i] * y[i + 1]) / a[i];
  }
  double spline(double x) {
    int p;
    p = int(x * n / len) + 1;
    double ans = 0;
    double h = len / n;
    double dx_1 = x - (len * (p - 1)) / n;
    double dx_2 = x - (len * p) / n;
    double frac_1 = dx_1 / h;
    double frac_2 = dx_2 / h;
    ans =
        ((1 + 2 * frac_1) * value[p - 1] + dx_1 * y[p - 1]) * frac_2 * frac_2 +
        ((1 - 2 * frac_2) * value[p] + dx_2 * y[p]) * frac_1 * frac_1;
    return ans;
  }
};