#include <stdio.h>

#define TILE 32

__global__ void sumCols(int* a, int* b, int n) {
    __shared__ int cache[TILE];

    int col = blockIdx.x;
    int tx = threadIdx.x;

    int sum = 0;
    for (int row = tx; row < n; row += TILE) {
        sum += a[row * n + col];
    }

    cache[tx] = sum;
    __syncthreads();

    for (int i = TILE / 2; i > 0; i >>= 1) {
        if (tx < i) {
            cache[tx] += cache[tx + i];
        }
        __syncthreads();
    }

    if (tx == 0) {
        b[col] = cache[0];
    }
}

int main() {
    const int n = 256;
    
    int* a = new int[n*n];
    int* b = new int[n];
    int *dev_a, *dev_b;
    cudaMalloc((void**)&dev_a, n * n* sizeof(int));
    cudaMalloc((void**)&dev_b, n * sizeof(int));

    for (int i=0; i<n*n; ++i) {
        a[i] = i + 1;
    }

    cudaMemcpy(dev_a, a, n * n * sizeof(int), cudaMemcpyHostToDevice);

    sumCols<<<n, TILE>>>(dev_a, dev_b, n);

    cudaMemcpy(b, dev_b, n * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(dev_a); cudaFree(dev_b);
    delete[] a; delete[] b;
    return 0;
}