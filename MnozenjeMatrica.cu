#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define TILE 2

__global__ void matmul(float* a, float* b, float* c, int m, int k, int n) {
    __shared__ float cacheA[TILE][TILE];
    __shared__ float cacheB[TILE][TILE];

    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;
    int ty = threadIdx.y; int tx = threadIdx.x;

    float sum = 0.0f;

    for (int phase = 0; phase < (k + TILE - 1) / TILE; phase++) {
        if (row < m && phase * TILE + tx < k) {
            cacheA[ty][tx] = a[row * k + phase * TILE + tx];
        } else {
            cacheA[ty][tx] = 0.0f;
        }

        if (col < n && phase * TILE + ty < k) {
            cacheB[ty][tx] = b[(phase * TILE + ty) * n + col];
        } else {
            cacheB[ty][tx] = 0.0f;
        }

        __syncthreads();

        for (int j=0; j<TILE; ++j) {
            sum += cacheA[ty][j] * cacheB[j][tx];
        }

        __syncthreads();
    }

    if (row < m && col < n) {
        c[row * n + col] = sum;
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
    const int k = 4;

    float* a = (float*)malloc(m * k * sizeof(float));
    float* b = (float*)malloc(k * n * sizeof(float));
    float* c = (float*)malloc(m * n * sizeof(float));

    srand(time(0));
    fillMat(a, m * n);
    fillMat(b, n * k);

    float *dev_a, *dev_b, *dev_c;
    cudaMalloc((void**)&dev_a, m * k * sizeof(float));
    cudaMalloc((void**)&dev_b, k * n * sizeof(float));
    cudaMalloc((void**)&dev_c, m * n * sizeof(float));

    cudaMemcpy(dev_a, a, m * k * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, b, k * n * sizeof(float), cudaMemcpyHostToDevice);

    dim3 blocksize(TILE, TILE);
    dim3 gridsize((n + TILE - 1) / TILE, (m + TILE - 1) / TILE);
    matmul<<<gridsize, blocksize>>>(dev_a, dev_b, dev_c, m, k, n);
    cudaDeviceSynchronize();

    cudaMemcpy(c, dev_c, m * n * sizeof(float), cudaMemcpyDeviceToHost);

    for (int i=0; i<m; ++i) {
        for (int j=0; j<n; ++j) {
            printf("%.2f ", c[i * n + j]);
        }
        printf("\n");
    }

    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);
    free(a);
    free(b);
    free(c);
    return 0;
}