#include <stdio.h>

#define imin(a, b) (a < b ? a : b)
#define sum_squares(x) (x * (x+1) * (2*x+1) / 6) // za proveru

#define n 256
#define BLOCKSIZE 256

const int GRIDSIZE = imin(32, (n + BLOCKSIZE - 1) / BLOCKSIZE);

__global__ void dotProd(double* a, double* b, double* c) {
    __shared__ double cache[BLOCKSIZE]; // dele sve niti u bloku, svaki blok ima novu kopiju
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int cacheIndex = threadIdx.x;
    double temp = 0;
    while (tid < n) {
        temp += a[tid] * b[tid];
        tid += blockDim.x * gridDim.x;
    }
    cache[cacheIndex] = temp;
    __syncthreads(); // kao barrier u mpi, ceka sve niti da dodju do ovde

    // podrazumeva se da je BLOCKSIZE stepen dvojke za redukciju
    int i = blockDim.x / 2;
    while (i != 0) {
        if (cacheIndex < i) {
            cache[cacheIndex] += cache[cacheIndex + i];
        }
        __syncthreads();
        i /= 2;
    }

    if (cacheIndex == 0) {
        c[blockIdx.x] = cache[0];
    }
}

int main() {
    double *a, *b, *partial_c;
    double *dev_a, *dev_b, *dev_partial_c;

    a = (double*)malloc(n * sizeof(double));
    b = (double*)malloc(n * sizeof(double));
    partial_c = (double*)malloc(GRIDSIZE * sizeof(double));

    cudaMalloc((void**)&dev_a, n * sizeof(double));
    cudaMalloc((void**)&dev_b, n * sizeof(double));
    cudaMalloc((void**)&dev_partial_c, GRIDSIZE * sizeof(double));

    for (int i=0; i<n; ++i) {
        a[i] = i;
        b[i] = i * 2;
    }

    cudaMemcpy(dev_a, a, n * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, b, n * sizeof(double), cudaMemcpyHostToDevice);
    dotProd<<<GRIDSIZE, BLOCKSIZE>>>(dev_a, dev_b, dev_partial_c);

    cudaMemcpy(partial_c, dev_partial_c, GRIDSIZE * sizeof(double), cudaMemcpyDeviceToHost);
    
    double c = 0;
    for (int i=0; i<GRIDSIZE; ++i) {
        c += partial_c[i];
    }
    
    printf("%.6g = %.6g?\n", c, 2 * sum_squares((double)(n - 1)));

    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_partial_c);

    free(a);
    free(b);
    free(partial_c);

    return 0;
}