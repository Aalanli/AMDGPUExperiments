#pragma once

#include "hip/hip_runtime.h"
#include <hip/amd_detail/amd_hip_runtime.h>
// #include <hip/amd_detail/amd_hip_vector_types.h>
#include <hip/hip_runtime_api.h>

#include <stdio.h>
#include <stdlib.h>

#ifndef HIP_ASSERT
#define HIP_ASSERT(error)                    \
    if(error != hipSuccess)                       \
    {                                             \
        fprintf(stderr,                           \
                "hip error: '%s'(%d) at %s:%d\n", \
                hipGetErrorString(error),         \
                error,                            \
                __FILE__,                         \
                __LINE__);                        \
        exit(EXIT_FAILURE);                       \
    }
#endif

#ifndef EXPORT
#define EXPORT extern "C" __attribute__((visibility("default")))
#endif

#define KERNEL(lb_) static __global__ __launch_bounds__((lb_)) void

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


template <typename F>
float bench(F&& func, int warmup, int iter) {
    hipEvent_t starts[iter];
    hipEvent_t   ends[iter];

    for (int i = 0; i < iter; i++) {
        hipEventCreate(starts + i);
        hipEventCreate(ends + i);
    }


    float* temp_buf;
    hipMalloc(&temp_buf, int(1e3));

    for (int i = 0; i < warmup; i++) {
        func();
    }

    hipDeviceSynchronize();
    for (int i = 0; i < iter; i++) {
        fill(temp_buf, 0.0f, int(1e3));
        hipEventRecord(starts[i]);

        func();
        hipEventRecord(ends[i]);

    }
    hipDeviceSynchronize();
    float times = 0.0;
    for (int i = 0; i < iter; i++) {
        float t;
        hipEventElapsedTime(&t, starts[i], ends[i]);
        times += t;
    }

    hipFree(temp_buf);

    for (int i = 0; i < iter; i++) {
        hipEventDestroy(starts[i]);
        hipEventDestroy(ends[i]);
    }

    return times / iter;
}

int cdiv(int a, int b) {
    return (a + b - 1) / b;
}