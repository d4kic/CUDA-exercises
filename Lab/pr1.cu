#include <stdio.h>

#define TILE 32

__global__ void computeB(float* a, float* b, int n) {
    __shared__ float shm_a[TILE + 2]; // +2 jer trebaju i+1, i+2
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int tx = threadIdx.x;

    if (i < n) {
        shm_a[tx] = a[i];
    }

    // prva dva thread-a ucitavaju a[i+1] i a[i+2] na kraj shared mem.
    if (tx < 2 && i + TILE < n ) {
        shm_a[tx + TILE] = a[i + TILE];
    }

    __syncthreads();

    if (i < n - 2) {
        b[i] = (shm_a[tx] + shm_a[tx + 1] + shm_a[tx + 2]) / 3.0f;
    }
}

int main() {
    const int n = 256;

    float* a = new float[n];
    float* b = new float[n - 2];
    
    for (int i=0; i<n; ++i) {
        a[i] = i + 1;
    }

    float *dev_a, *dev_b;
    cudaMalloc(&dev_a, n * sizeof(float));
    cudaMalloc(&dev_b, (n - 2) * sizeof(float));

    cudaMemcpy(dev_a, a, n * sizeof(int), cudaMemcpyHostToDevice);

    const int BLOCKS = (n + TILE - 1) / TILE;
    computeB<<<BLOCKS, TILES>>>(dev_a, dev_b, n);

    cudaMemcpy(dev_b, b, (n - 2) * sizeof(int), cudaMemcpyDeviceToHost);

    for (int i = 0; i < n - 2; ++i) {
        printf("%d ", b[i]);
    }

    cudaFree(dev_a);
    cudaFree(dev_b);
    delete[] a;
    delete[] b;
    return 0;
}