#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define TILE 2

__global__ void transp(float* a, float* b, int m, int n) {
    __shared__ float tile[TILE][TILE + 1]; // +1 za izbegavanje bank konflikta
    int colA = blockIdx.x * TILE + threadIdx.x;
    int rowA = blockIdx.y * TILE + threadIdx.y;

    if (rowA < m && colA < n) {
        tile[threadIdx.y][threadIdx.x] = a[rowA * n + colA];
    }

    __syncthreads();

    int colB = blockIdx.y * TILE + threadIdx.x;
    int rowB = blockIdx.x * TILE + threadIdx.y;

    if (rowB < n && colB < m) {
        b[rowB * m + colB] = tile[threadIdx.x][threadIdx.y];
    }
}

__host__ void fillMat(float* mat, int size) {
    for (int i=0; i<size; ++i) {
        mat[i] = (float)rand() / RAND_MAX;
    }
}

int main() {
    const int n = 4;
    const int m = 4;

    float* a = (float*)malloc(m * n * sizeof(float));
    float* b = (float*)malloc(m * n * sizeof(float));

    srand(time(0));
    fillMat(a, m * n);

    float *dev_a, *dev_b;
    cudaMalloc((void**)&dev_a, m * n * sizeof(float));
    cudaMalloc((void**)&dev_b, m * n * sizeof(float));

    cudaMemcpy(dev_a, a, m * n * sizeof(float), cudaMemcpyHostToDevice);

    dim3 blocksize(TILE, TILE);
    dim3 gridsize((n + TILE - 1) / TILE, (m + TILE - 1) / TILE);
    transp<<<gridsize, blocksize>>>(dev_a, dev_b, m, n);
    cudaDeviceSynchronize();

    cudaMemcpy(b, dev_b, m * n * sizeof(float), cudaMemcpyDeviceToHost);

    printf("A:\n");
    for (int i=0; i<m; ++i) {
        for (int j=0; j<n; ++j) {
            printf("%.2f ", a[i * n + j]);
        }
        printf("\n");
    }

    printf("B:\n");
    for (int i=0; i<n; ++i) {
        for (int j=0; j<m; ++j) {
            printf("%.2f ", b[i * m + j]);
        }
        printf("\n");
    }

    cudaFree(dev_a);
    cudaFree(dev_b);
    free(a);
    free(b);
    return 0;
}