#include <stdio.h>

#define BLOCKSIZE 32

__global__ void reduce(double *a, double *b, int n) {
    __shared__ double shmem[BLOCKSIZE + 2];
    int i = threadIdx.x + blockIdx.x * blockDim.x;

    if (i < n) {
        shmem[threadIdx.x] = a[i];
    }

    if (threadIdx.x < 2 && i + BLOCKSIZE < n) {
        shmem[threadIdx.x + BLOCKSIZE] = a[i + BLOCKSIZE];
    }

    __syncthreads();

    if (i < n - 2) {
        b[i] = (shmem[threadIdx.x] * shmem[threadIdx.x + 1] * shmem[threadIdx.x + 2]) / 200;
    }
} 

int main() {
    const int n = 256;
    double *a, *b;
    double *dev_a, *dev_b;

    a = new double[n];
    b = new double[n - 2];
    cudaMalloc((void**)&dev_a, n * sizeof(double));
    cudaMalloc((void**)&dev_b, (n - 2) * sizeof(double));

    for (int i=0; i<n; ++i) {
        a[i] = i + 1;
    }

    cudaMemcpy(dev_a, a, n * sizeof(double), cudaMemcpyHostToDevice);

    const int GRIDSIZE = (n + BLOCKSIZE - 1) / BLOCKSIZE;
    reduce<<<GRIDSIZE, BLOCKSIZE>>>(dev_a, dev_b, n);

    cudaMemcpy(b, dev_b, (n-2) * sizeof(double), cudaMemcpyDeviceToHost);

    for (int i=0; i<n-2; ++i) {
        printf("%d ", b[i]);
    }

    cudaFree(dev_a); cudaFree(dev_b);
    delete[] a; delete[] b;
    return 0;
}