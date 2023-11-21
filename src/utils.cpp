#include "hip_utils.hpp"

void __global__ init_kernel(float* __restrict__ a, float v, int n) {
    const int stride = blockDim.x * gridDim.x;
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    while (tid < n) {
        a[tid] = v;
        tid += stride;
    }
}

void fill(float* __restrict__ a, float v, int n) {
    int n_grid = (n + 1023) / 1024;
    hipLaunchKernelGGL(init_kernel, dim3(n_grid), dim3(1024), 0, 0, a, v, n);
}
