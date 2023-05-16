#include <stdio.h>

#define N 10

__global__ void add(int *a, int *b, int *c)
{
    int tid = blockIdx.x; // this thread handles the data at its thread id
    if (tid < N)
    {
        c[tid] = a[tid] + b[tid];
    }
    printf("%d", c[tid]);
}

int main(void)
{
    int a[N], b[N], c[N];
    int *dev_a, *dev_b, *dev_c;

    // allocate the memory on the GPU
    cudaMalloc((void **)&dev_a, N * sizeof(int));
    cudaMalloc((void **)&dev_b, N * sizeof(int));
    cudaMalloc((void **)&dev_c, N * sizeof(int));

    // fill the arrays 'a' and 'b' on the CPU
    for (int i = 0; i < N; i++)
    {
        a[i] = -i;
        b[i] = i * i;
    }

    // copy the arrays 'a' and 'b' to the GPU
    cudaMemcpy(dev_a, a, N * sizeof(int),
               cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, b, N * sizeof(int),
               cudaMemcpyHostToDevice);

    add<<<N, 1>>>(dev_a, dev_b, dev_c);
    cudaError_t error = cudaGetLastError();
    printf("CUDA error: %s\n", cudaGetErrorString(error));

    // copy the array 'c' back from the GPU to the CPU
    cudaMemcpy(c, dev_c, N * sizeof(int),
               cudaMemcpyDeviceToHost);

    // display the results
    for (int i = 0; i < N; i++)
    {
        printf("%d + %d = %d\n", a[i], b[i], c[i]);
    }

    // free the memory allocated on the GPU
    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);
    return 0;
}
