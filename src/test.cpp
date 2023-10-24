#include <assert.h>
#include <bits/floatn-common.h>
#include <stdio.h>
#include <algorithm>
#include <stdlib.h>
#include <iostream>
#include "hip/hip_runtime.h"

extern "C" void saxpy(float* a, float* b, float* c, int n, int d);

__global__ void saxpy_kernel(
    const float* __restrict__ a,
    const float* __restrict__ b,
    float* __restrict__ c,
    int n, int d) 
{
    int bid = hipBlockIdx_y;
    int tid = hipThreadIdx_x + hipBlockIdx_x * hipBlockDim_x;
    int stride = hipBlockDim_x * hipGridDim_x;

    while (tid < n) {
        c[bid * d + tid] = a[bid * d + tid] + b[bid * d + tid];
        tid += stride;
    }
}

#ifndef Test34
#define Test34 34
#endif

void saxpy(float* a, float* b, float* c, int n, int d) {
    
    #ifdef Test34
    hipLaunchKernelGGL(saxpy_kernel, dim3(n / 256, d), dim3(256), 0, 0, a, b, c, n, d);
    #endif
}

int main() {
    #ifdef Test34
    std::cout << Test34 << std::endl;
    #endif
}