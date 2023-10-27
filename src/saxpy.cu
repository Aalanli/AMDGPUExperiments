#include <stdio.h>

#ifndef LAUNCH_NAME
#define LAUNCH_NAME saxpy
#endif

#ifndef BLOCKSIZE
#define BLOCKSIZE 256
#endif

#ifndef REPEATS
#define REPEATS 4
#endif

__global__ void saxpy_kernel(
    const float* __restrict__ a,
    const float* __restrict__ x,
    const float* __restrict__ b,
    float* __restrict__ c,
    int n) 
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;

    while (tid < n) {
        c[tid] = a[tid] * x[tid] + b[tid];
        tid += stride;
    }
}

extern "C" bool LAUNCH_NAME(float* a, float* x, float* b, float* c, int n);

bool LAUNCH_NAME(float* a, float* x, float* b, float* c, int n) {
    const int repeats = BLOCKSIZE * REPEATS;
    saxpy_kernel<<<(n + repeats - 1) / repeats, BLOCKSIZE>>>(a, x, b, c, n);
    auto err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
        return false;
    }

    return true;
}

