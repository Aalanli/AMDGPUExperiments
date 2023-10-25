#include "hip/hip_runtime.h"
#include <hip/amd_detail/amd_hip_runtime.h>
#include <hip/amd_detail/amd_hip_vector_types.h>

#include <cassert>
#include <hip/hip_runtime_api.h>
#include <iostream>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string>

#include "utils.hpp"

void __global__ copy_kernel(float* __restrict__ a, float* __restrict__ b, const int n) {
    const int stride = blockDim.x * gridDim.x;
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    while (tid < n) {
        b[tid] = a[tid];
        tid += stride;
    }
}

void __global__ copy_kernel_pipeline(const float* __restrict__ a, float* __restrict__ b, const int n) {
    const int stride = blockDim.x * gridDim.x;
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int i = 0;

    float locals[2];
    
    if (tid < n) {
        locals[0] = a[tid];
    }

    while (tid < n) {
        b[tid] = locals[i];
        tid += stride;
        locals[(i + 1) % 2] = tid < n ? a[tid] : 0.0f;
        i += 1;
    }
}

void __global__ copy_kernelf4(const float4* __restrict__ a, float4* __restrict__ b, const int n) {
    const int stride = blockDim.x * gridDim.x;
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    while (tid < n) {
        b[tid] = a[tid];
        tid += stride;
    }
}

int main(int argc, char** argv) {
    const int repeats = 1;
    assert(argc == 1);
    const int n = std::stoi(argv[1]);
    printf("copying %i floats\n", n);

    HIP_ASSERT(hipDeviceSynchronize());
    float* a;
    float* b;

    HIP_ASSERT(hipMalloc((void**)&a, n * sizeof(float)));
    HIP_ASSERT(hipMalloc((void**)&b, n * sizeof(float)));

    int grid_dim = (n + 1023) / (1024 * repeats);
    hipLaunchKernelGGL(copy_kernel, dim3(grid_dim), dim3(1024), 0, 0, a, b, n);
    hipLaunchKernelGGL(copy_kernel_pipeline, dim3(grid_dim), dim3(1024), 0, 0, a, b, n);

    if (n % 4 == 0) {
        grid_dim = (n / 4 + 1023) / (1024 * repeats);
        hipLaunchKernelGGL(copy_kernelf4, dim3(grid_dim), dim3(1024), 0, 0, (float4*) a, (float4*) b, n / 4);
    }

    hipDeviceSynchronize();

    hipFree(a);
    hipFree(b);
}