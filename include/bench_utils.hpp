#pragma once
#include "hip_utils.hpp"

void fill(float* __restrict__ a, float v, int n);

template <typename F>
float inline bench(F&& func, int warmup, int iter) {
    hipEvent_t starts[iter];
    hipEvent_t   ends[iter];

    for (int i = 0; i < iter; i++) {
        HIP_ASSERT(hipEventCreate(starts + i));
        HIP_ASSERT(hipEventCreate(ends + i));
    }


    float* temp_buf;
    HIP_ASSERT(hipMalloc(&temp_buf, int(1e3)));

    for (int i = 0; i < warmup; i++) {
        func();
    }

    HIP_ASSERT(hipDeviceSynchronize());
    for (int i = 0; i < iter; i++) {
        fill(temp_buf, 0.0f, int(1e3));
        HIP_ASSERT(hipEventRecord(starts[i]));

        func();
        HIP_ASSERT(hipEventRecord(ends[i]));

    }
    HIP_ASSERT(hipDeviceSynchronize());
    float times = 0.0;
    for (int i = 0; i < iter; i++) {
        float t;
        HIP_ASSERT(hipEventElapsedTime(&t, starts[i], ends[i]));
        times += t;
    }

    HIP_ASSERT(hipFree(temp_buf));

    for (int i = 0; i < iter; i++) {
        HIP_ASSERT(hipEventDestroy(starts[i]));
        HIP_ASSERT(hipEventDestroy(ends[i]));
    }

    return times / iter;
}