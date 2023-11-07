#include "hip_utils.hpp"
#include <hip/amd_detail/amd_hip_runtime.h>


/// computes #iters fma operations
/// simulates compute bound arithmetic intense workload
__device__ float arith_workload(float v, int iters) {
    for (int i = 0; i < iters; ++i) {
        v = v * v + v;
    }
    return v;
}


template <int NPipeline>
__global__ void register_pipeline(float* __restrict__ a, float* __restrict__ b, int n, int iters) {
    float pipeline[NPipeline];

    const int tid = threadIdx.x + blockIdx.x * blockDim.x;
    const int stride = blockDim.x * gridDim.x;
    for (int i = 0; i < NPipeline - 1; ++i) {
        int offset = tid + i * stride;
        pipeline[i] = offset < n ? a[tid + i * stride] : 0.0f;
    }

    float res = 0.0f;
    for (int i = 0; i < cdiv(n, stride); i++) {
        // compute
        int offset = tid + i * stride;
        if (offset < n)
            res += arith_workload(pipeline[i % NPipeline], iters);
        // load next iteration
        int next_offset = offset + (NPipeline - 1) * stride;
        if (next_offset < n)
            pipeline[(i + NPipeline - 1) % NPipeline] = a[next_offset];
    }
    b[tid] = res;
}

/// there are two possible configurations of pipelining for this kernel
/// as it simulates the simt gemm kernel, where global -> register -> shared -> register
/// 1. the first pipeline approach is to pipeline across registers
/// 2. the second pipeline approach is to pipeline across shared memory
template <int NPipeline>
__global__ void shared_mem_regs_pipeline(float* __restrict__ a, float* __restrict__ b, int n, int iters) {
    extern __shared__ float smem[];
    float pipeline[NPipeline];

    const int tid = threadIdx.x + blockIdx.x * blockDim.x;
    const int stride = blockDim.x * gridDim.x;
    for (int i = 0; i < NPipeline - 1; ++i) {
        int offset = tid + i * stride;
        pipeline[i] = offset < n ? a[tid + i * stride] : 0.0f;
    }

    float res = 0.0f;
    for (int i = 0; i < cdiv(n, stride); i++) {
        // compute
        int offset = tid + i * stride;
        float v = pipeline[i % NPipeline];
        smem[threadIdx.x] = v;
        __syncthreads();
        v = smem[blockDim.x - 1 - threadIdx.x];
        res += arith_workload(v, iters);
        __syncthreads();
        
        // load next iteration
        int next_offset = offset + (NPipeline - 1) * stride;
        if (next_offset < n)
            pipeline[(i + NPipeline - 1) % NPipeline] = a[next_offset];
    }
    b[tid] = res;
}

template <int NPipeline>
__global__ void shared_mem_smem_pipeline(float* __restrict__ a, float* __restrict__ b, int n, int iters) {
    extern __shared__ float smem[];
    float pipeline[NPipeline];

    const int tid = threadIdx.x + blockIdx.x * blockDim.x;
    const int stride = blockDim.x * gridDim.x;
    for (int i = 0; i < NPipeline - 1; ++i) {
        int offset = tid + i * stride;
        smem[(blockDim.x - 1 - threadIdx.x) + i * blockDim.x] = offset < n ? a[offset] : 0.0f;
    }
    __syncthreads();

    float res = 0.0f;
    for (int i = 0; i < cdiv(n, stride); i++) {
        // compute
        int offset = tid + i * stride;
        float v = smem[threadIdx.x + (i % NPipeline) * blockDim.x];
        res += arith_workload(v, iters);
        
        // load next iteration
        int next_offset = offset + (NPipeline - 1) * stride;
        if (next_offset < n)
            smem[((i + NPipeline - 1) % NPipeline) * blockDim.x + threadIdx.x] = a[next_offset];
        __syncthreads();
    }
    b[tid] = res;
}

