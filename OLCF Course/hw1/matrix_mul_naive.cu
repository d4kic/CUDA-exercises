#include <stdio.h>

#define BSIZE 16

const int n = 5;
const int m = 4;
const int k = 6;

__global__ void matmul(int* a, int* b, int* c) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < n && col < k) {
        int tmp = 0;
        for (int i=0; i<m; ++i) {
            tmp += a[row * m + i] * b[i * k + col];
        }
        c[row * k + col] = tmp;
    }
}

int main() {
    int *a, *b, *c;
    int *dev_a, *dev_b, *dev_c;

    a = (int*)malloc(n * m * sizeof(int));
    b = (int*)malloc(m * k * sizeof(int));
    c = (int*)malloc(n * k * sizeof(int));
    cudaMalloc((void**)&dev_a, n * m * sizeof(int));
    cudaMalloc((void**)&dev_b, m * k * sizeof(int));
    cudaMalloc((void**)&dev_c, n * k * sizeof(int));

    for (int i=0; i<n; ++i) {
        for (int j=0; j<m; ++j) {
            a[i * m + j] = i + j + 1;
        }
    }

    for (int i=0; i<m; ++i) {
        for (int j=0; j<k; ++j) {
            b[i * k + j] = i + j + 1;
        }
    }

    cudaMemcpy(dev_a, a, n * m * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, b, m * k * sizeof(int), cudaMemcpyHostToDevice);

    dim3 BLOCKSIZE(BSIZE, BSIZE);
    dim3 GRIDSIZE((k + BSIZE - 1) / BSIZE, (n + BSIZE - 1) / BSIZE);
    matmul<<<GRIDSIZE, BLOCKSIZE>>>(dev_a, dev_b, dev_c);
    cudaDeviceSynchronize();

    cudaMemcpy(c, dev_c, n * k * sizeof(int), cudaMemcpyDeviceToHost);

    for (int i=0; i<n; ++i) {
        for (int j=0; j<k; ++j) {
            printf("%d ", c[i * k + j]);
        }
        printf("\n");
    }

    cudaFree(dev_a); cudaFree(dev_b); cudaFree(dev_c);
    free(a); free(b); free(c);
    return 0;
}