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

template <int Iter>
__device__ float arith_workload(float v) {
    #pragma unroll
    for (int i = 0; i < Iter; ++i) {
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

#ifndef SHARED_ITERS
#define SHARED_ITERS 1
#endif

#ifndef REG_ITERS
#define REG_ITERS 1
#endif

#ifndef BLOCK_DIM
#define BLOCK_DIM 256
#endif

#ifndef NPIPELINE_SMEM
#define NPIPELINE_SMEM 1
#endif

#ifndef NPIPELINE_REGS
#define NPIPELINE_REGS 1
#endif

// hypothesis: more bank conflicts requires more NPIPELINE_REGS
#ifndef SMEM_READ_STRIDE 
#define SMEM_READ_STRIDE 1 // no bank conflict
#endif 

#ifndef SMEM_STORE_STRIDE
#define SMEM_STORE_STRIDE 1 // no bank conflict
#endif


__global__ void __launch_bounds__(BLOCK_DIM) shared_smem_reg_pipelinev1(
    float* __restrict__ a, float* __restrict__ b, int n
) {
    extern __shared__ float smem[];
    const int tid = threadIdx.x + blockIdx.x * blockDim.x;
    const int stride = blockDim.x * gridDim.x;
    for (int i = 0; i < NPIPELINE_SMEM; ++i) {
        int offset = tid + i * stride;
        smem[(threadIdx.x * SMEM_STORE_STRIDE) % BLOCK_DIM + i * BLOCK_DIM] = offset < n ? a[offset] : 0.0f;
    }
    __syncthreads();
    float regs_pipeline[NPIPELINE_REGS];
    float res = 0.0f;

    int ntiles = (n + stride - 1) / stride - 1;
    for (int i = 0; i < ntiles; i++) {
        // prefetch into registers
        static_assert(SHARED_ITERS % NPIPELINE_REGS == 0 && SHARED_ITERS > 0, "");
        const int smem_slice = (i % NPIPELINE_SMEM) * BLOCK_DIM;
        #pragma unroll
        for (int j = 0; j < NPIPELINE_REGS; ++j) {
            regs_pipeline[j] = smem[((threadIdx.x / SMEM_READ_STRIDE) + j) % BLOCK_DIM + smem_slice];
        }
        // simulate smem workload
        #pragma unroll
        for (int j = 0; j < SHARED_ITERS - 1; ++j) {
            res += arith_workload<REG_ITERS>(regs_pipeline[j % NPIPELINE_REGS]);
            regs_pipeline[(j + NPIPELINE_REGS - 1) % NPIPELINE_REGS] = smem[(threadIdx.x / SMEM_READ_STRIDE + j + NPIPELINE_REGS - 1) % BLOCK_DIM + smem_slice];
        }
        res += arith_workload<REG_ITERS>(regs_pipeline[(SHARED_ITERS - 1) % NPIPELINE_REGS]);

        // load next iteration
        int next_offset = tid + (i + NPIPELINE_SMEM - 1) * stride;
        smem[(threadIdx.x * SMEM_STORE_STRIDE) % BLOCK_DIM + ((i + NPIPELINE_SMEM - 1) % NPIPELINE_SMEM) * BLOCK_DIM] = next_offset < n ? a[next_offset] : 0.0f;
        __syncthreads();
    }

    // last tile
    int offset = tid + ntiles * stride;
    const int smem_slice = (ntiles % NPIPELINE_SMEM) * BLOCK_DIM;
    #pragma unroll
    for (int j = 0; j < NPIPELINE_REGS; ++j) {
        regs_pipeline[j] = smem[((threadIdx.x / SMEM_READ_STRIDE) + j) % BLOCK_DIM + smem_slice];
    }
    #pragma unroll
    for (int j = 0; j < SHARED_ITERS - 1; ++j) {
        res += arith_workload<REG_ITERS>(regs_pipeline[j % NPIPELINE_REGS]);
        regs_pipeline[(j + NPIPELINE_REGS - 1) % NPIPELINE_REGS] = smem[(threadIdx.x / SMEM_READ_STRIDE + j + NPIPELINE_REGS - 1) % BLOCK_DIM + smem_slice];
    }
    res += arith_workload<REG_ITERS>(regs_pipeline[(SHARED_ITERS - 1) % NPIPELINE_REGS]);

    // store
    if (tid < n)
        b[tid] = res;
}


#ifndef LAUNCH_NAME
#define LAUNCH_NAME pipeline_test
#endif


EXPORT bool LAUNCH_NAME(float* a, float* b, int grid_dim, int repeats, int additional_smem) {
    // float* a;
    // float* b;
    // hipMalloc(&a, sizeof(float) * grid_dim * BLOCK_DIM * repeats);
    // hipMalloc(&b, sizeof(float) * grid_dim * BLOCK_DIM);

    int smem = BLOCK_DIM * NPIPELINE_SMEM * sizeof(float) + additional_smem;
    hipLaunchKernelGGL(shared_smem_reg_pipelinev1, dim3(grid_dim), dim3(BLOCK_DIM), smem, 0, a, b, grid_dim * BLOCK_DIM * repeats);

    // hipFree(a);
    // hipFree(b);
    return true;
}