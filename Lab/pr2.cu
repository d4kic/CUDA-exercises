#include <stdio.h>

#define TILE 32

__global__ void colSum(float* a, float* b, int n) {
    __shared__ float shmem[TILE];

    int col = blockIdx.x;
    int tx = threadIdx.x;

    float sum = 0.0f;
    for (int row = tx; row < n; row += TILE) {
        sum += a[row * n + col];
    }

    shmem[tx] = sum;
    __syncthreads();

    // redukcija
    for (int i = TILE / 2; i > 0; i >>= 1) { // ovo >>= je isto kao i /= 2
        if (tx < i) {
            shmem[tx] += shmem[tx + i];
        }
        __syncthreads();
    }

    if (tx == 0) {
        b[col] = shmem[0];
    }
}

int main() {
    const int n = 256;

    float* a = new float[n * n];
    float* b = new float[n];
    
    for (int i = 0; i < n * n; ++i) {
        a[i] = i + 1;
    }

    float *dev_a, *dev_b;
    cudaMalloc(&dev_a, n * n * sizeof(float));
    cudaMalloc(&dev_b, n * sizeof(float));

    cudaMemcpy(dev_a, a, n * n * sizeof(int), cudaMemcpyHostToDevice);

    colSum<<<n, TILES>>>(dev_a, dev_b, n);

    cudaMemcpy(dev_b, b, n * sizeof(int), cudaMemcpyDeviceToHost);

    for (int i = 0; i < n; ++i) {
        printf("%d ", b[i]);
    }

    cudaFree(dev_a);
    cudaFree(dev_b);
    delete[] a;
    delete[] b;
    return 0;
}
