#include <stdio.h>

#define n 4096
#define RADIUS 3
#define BLOCKSIZE 16

// u sustini za svaki element niza se racuna zbir sa njegovim susedima u okviru [-RADIUS, RADIUS]
// ovde je RADIUS = 3 tako da out[i] = in[i-3] + in[i-2] + in[i-1] + in[i] + in[i+1] + in[i+2] + in[i+3]
__global__ void stencil(int *in, int *out) {

}

int main() {
    int *in, *out;
    int *d_in, *d_out;
    
}