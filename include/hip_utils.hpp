#pragma once

#include "hip/hip_runtime.h"
#include <hip/amd_detail/amd_hip_runtime.h>
// #include <hip/amd_detail/amd_hip_vector_types.h>
#include <hip/hip_runtime_api.h>

#include <stdio.h>
#include <stdlib.h>


#ifndef EXPORT
#define EXPORT extern "C" __attribute__((visibility("default")))
#endif

#define KERNEL(lb_) static __global__ __launch_bounds__((lb_)) void


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

__device__ __host__ int inline cdiv(int a, int b) {
    return (a + b - 1) / b;
}