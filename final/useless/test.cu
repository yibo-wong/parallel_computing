#include <cstdio>
#include <iostream>
using namespace std;

void read_data(double *mat) {
  for (int i = 0; i < 2500; i++)
    for (int j = 0; j < 2500; j++) mat[i * 2500 + j] = (i + 1) * (j + 1);
}

__global__ void add(double *a, double *b, double *c, double *d_sum, int nelem) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  extern __shared__ float shared_num[];
  float my_add = threadIdx.x == 0 ? float(blockIdx.x) : 0.0001;
  // float my_add = 0.0031415926;
  if (index < nelem) {
    for (int j = 0; j < 10; j++)
      for (int i = 0; i < 10000; i++) c[index] = a[index] * b[index] * i * j;
    if (!(index % 10000))
      printf("On gpu %d, %f * %f * 9999 = %f \n", index, a[index], b[index],
             c[index]);
  } else {
    // printf("On GPU %d, do not calc\n", index);
  }
  __syncthreads();
  atomicAdd(&shared_num[0], my_add);
  __syncthreads();
  if (!(index % 10000))
    printf("On GPU %d, blockIdx.x = %d, shared_num = %f\n", index, blockIdx.x,
           shared_num);
  __syncthreads();
  if (!threadIdx.x) d_sum[blockIdx.x] = double(shared_num[0]);
}

int main() {
  int nelem = 2500 * 2500;
  size_t nbytes = nelem * sizeof(double);
  double *host_a = new double[nelem];
  double *host_b = new double[nelem];
  double *host_c = new double[nelem];
  double *host_sum = new double[150];
  read_data(host_a);
  read_data(host_b);

  double *dev_a;
  double *dev_b;
  double *dev_c;
  double *dev_sum;
  cudaMalloc((double **)&dev_a, nbytes);
  cudaMalloc((double **)&dev_b, nbytes);
  cudaMalloc((double **)&dev_c, nbytes);
  cudaMalloc((double **)&dev_sum, 150 * sizeof(double));
  cudaMemcpy(dev_a, host_a, nbytes, cudaMemcpyHostToDevice);
  cudaMemcpy(dev_b, host_b, nbytes, cudaMemcpyHostToDevice);
  dim3 blk(200, 1, 1);
  dim3 thd(1000, 1, 1);
  cout << "here" << endl;
  add<<<blk, thd, 1 * sizeof(float)>>>(dev_a, dev_b, dev_c, dev_sum, nelem);
  cudaDeviceSynchronize();
  cudaError_t error = cudaGetLastError();
  printf("CUDA error: %s\n", cudaGetErrorString(error));
  cudaMemcpy(host_c, dev_c, nbytes, cudaMemcpyDeviceToHost);
  cudaMemcpy(host_sum, dev_sum, 150 * sizeof(double), cudaMemcpyDeviceToHost);
  for (int i = 0; i < 150; i++) {
    cout << i << ": " << host_sum[i] << endl;
  }
  cout << endl;
  cudaDeviceReset();
  return 0;
}