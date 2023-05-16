#include <math.h>
#include <stdio.h>

extern "C" void cuda_solve(double *f_1d, double *u_old, double *u_new, double eps, double r1, double r2, double r3, double r, int M, int N);

int N = 50;
int M = 50;
int BLOCK_DIM = 50;
int max_iter = 100;

__global__ void cuda_Jacobi(double *f_1d, double *u_old, double *u_new, double r1, double r2, double r3, double r)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    double resid = 0;
    if (j > 0 && j < N - 1 && i > 0 && i < M - 1)
    {
        resid =
            f_1d[j * M + i] -
            (r1 * (u_old[(j - 1) * M + i - 1] + u_old[(j - 1) * M + i + 1]) +
             r3 * (u_old[(j)*M + i - 1] + u_old[(j)*M + i + 1]) +
             r1 * (u_old[(j + 1) * M + i - 1] + u_old[(j + 1) * M + i + 1]) +
             r2 * (u_old[(j - 1) * M + i] + u_old[(j + 1) * M + i]) +
             r * u_old[j * M + i]);
        u_new[j * M + i] = u_new[j * M + i] + resid / r;
    }
}

void cuda_solve(double *f_1d, double *u_old, double *u_new, double eps, double r1, double r2, double r3, double r, int M, int N)
{
    int size = (M * N) * sizeof(double);
    double *cuda_f_1d, *cuda_u_old, *cuda_u_new;
    cudaEvent_t start, stop;
    float ker_time = 0;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaMalloc((void **)&cuda_f_1d, size);
    cudaMalloc((void **)&cuda_u_old, size);
    cudaMalloc((void **)&cuda_u_new, size);
    cudaMemcpy(cuda_f_1d, f_1d, size, cudaMemcpyHostToDevice);
    cudaMemcpy(cuda_u_old, u_old, size, cudaMemcpyHostToDevice);
    cudaMemcpy(cuda_u_new, u_new, size, cudaMemcpyHostToDevice);
    dim3 grid_dim(ceil(M / (double)(BLOCK_DIM)), ceil(N / (double)(BLOCK_DIM)));
    dim3 block_dim(BLOCK_DIM, BLOCK_DIM);
    int k = 0;
    cudaEventRecord(start, 0);
    while (k < max_iter)
    {
    }
}