#include <stdio.h>

#define n 10
#define BLOCKSIZE 32

const int GRIDSIZE = (n + BLOCKSIZE - 1) / BLOCKSIZE;

__global__ void add(int* a, int* b, int* c) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    while (tid < n) {
        c[tid] = a[tid] + b[tid];
        tid += blockDim.x * gridDim.x;
    }
}

int main() {
    int a[n], b[n], c[n];
    int *dev_a, *dev_b, *dev_c;

    cudaMalloc((void**)&dev_a, sizeof(int));
    cudaMalloc((void**)&dev_b, sizeof(int));
    cudaMalloc((void**)&dev_c, sizeof(int));

    for (int i=0; i<n; ++i) {
        a[i] = -i;
        b[i] = i * i;
    }

    cudaMemcpy(dev_a, a, n * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, b, n * sizeof(int), cudaMemcpyHostToDevice);

    add<<<GRIDSIZE, BLOCKSIZE>>>(dev_a, dev_b, dev_c);
    cudaMemcpy(&c, dev_c, sizeof(int), cudaMemcpyDeviceToHost);
    
    for (int i=0; i<n; ++i) {
        printf("%d + %d = %d\n", a[i], b[i], c[i]);
    }

    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);
    return 0;
}