#include <assert.h>
#include <bits/floatn-common.h>
#include <stdio.h>
#include <algorithm>
#include <stdlib.h>
#include <iostream>
#include "hip/hip_runtime.h"

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
    const float* __restrict__ b,
    float* __restrict__ c,
    int n, int d) 
{
    int bid = hipBlockIdx_y;
    int tid = hipThreadIdx_x + hipBlockIdx_x * hipBlockDim_x;
    int stride = hipBlockDim_x * hipGridDim_x;

    while (tid < d) {
        c[bid * d + tid] = a[bid * d + tid] + b[bid * d + tid];
        tid += stride;
    }
}

extern "C" void LAUNCH_NAME(float* a, float* b, float* c, int n, int d);

void LAUNCH_NAME(float* a, float* b, float* c, int n, int d) {
    hipLaunchKernelGGL(saxpy_kernel, dim3(d / BLOCKSIZE / REPEATS, n), dim3(BLOCKSIZE), 0, 0, a, b, c, n, d);
}

