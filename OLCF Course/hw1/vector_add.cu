#include <stdio.h>

#define N 512
#define BLOCKSIZE 32

__global__ void vector_add(int* a, int* b, int* c, int n) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}

int main() {
    int *a, *b, *c;
    int *dev_a, *dev_b, *dev_c;
    size_t size = N * sizeof(int);

    a = (int*)malloc(size);
    b = (int*)malloc(size);
    c = (int*)malloc(size);
    cudaMalloc((void**)&dev_a, size);
    cudaMalloc((void**)&dev_b, size);
    cudaMalloc((void**)&dev_c, size);

    for (int i=0; i<N; ++i) {
        a[i] = i + 1;
        b[i] = i * i;
    }

    cudaMemcpy(dev_a, a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, b, size, cudaMemcpyHostToDevice);

    vector_add<<<(N + BLOCKSIZE - 1) / BLOCKSIZE, BLOCKSIZE>>>(dev_a, dev_b, dev_c, N);

    cudaMemcpy(c, dev_c, size, cudaMemcpyDeviceToHost);

    for (int i=0; i<N; ++i) {
        printf("%d ", c[i]);
    }

    free(a); free(b); free(c);
    cudaFree(dev_a); cudaFree(dev_b); cudaFree(dev_c);
    return 0;
}