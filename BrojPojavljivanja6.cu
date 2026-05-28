#include <stdio.h>
#include <time.h>

#define n 16
#define BLOCKSIZE 4
// 1 blok u gridu, svaka nit ispituje 4 el

__device__ int compare(int a, int b) {
    if (a == b) {
        return 1;
    }
    return 0;
}

__global__ void kernelFn(int* d_in, int* d_out) {
    d_out[threadIdx.x] = 0;
    for (int i = 0; i < n / BLOCKSIZE; ++i) {
        int val = d_in[i * BLOCKSIZE + threadIdx.x]; // ide 0, 4, 8, 12 za nit 0, 1, 5, 9, 13 za nit 1 itd.
        d_out[threadIdx.x] += compare(val, 6);
    }
}

__host__ void prebaciNaDev(int* in, int* out) {
    int *d_in, *d_out;

    cudaMalloc((void**)&d_in, n * sizeof(int));
    cudaMalloc((void**)&d_out, BLOCKSIZE * sizeof(int));
    
    cudaMemcpy(d_in, in, n * sizeof(int), cudaMemcpyHostToDevice);

    kernelFn<<<1, BLOCKSIZE>>>(d_in, d_out);
    cudaThreadSynchronize();

    cudaMemcpy(d_out, out, BLOCKSIZE * sizeof(int), cudaMemcpyDeviceToHost);
}

int main() {
    srand(time(0));
    int *in, *out;

    prebaciNaDev(in, out);

    int sum = 0;
    for (int i = 0; i < BLOCKSIZE; ++i) {
        sum += out[i];
    }
    printf("Res = %d\n", sum);

    return 0;
}