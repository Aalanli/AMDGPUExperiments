#include "hip_utils.hpp"
#include <hip/amd_detail/amd_hip_runtime.h>

#include "layouts.cpp"
#include "block_tiles.cpp"
#include "kernel_tiles.cpp"


template <typename GemmInstance>
__global__ __launch_bounds__(GemmInstance::nthreads()) void gemm_kernel(
    const float * __restrict__ A,
    const float * __restrict__ B,
    float * __restrict__ C,
    int M, int K, int N
) {
    __shared__ char smem[GemmInstance::used_smem_bytes()];
    auto inst = GemmInstance((float*) smem, A, B, C, M, K, N);
    inst.run();
}

template <typename GemmInstance>
__global__ __launch_bounds__(GemmInstance::nthreads()) void gemm_kernel_dyn_smem(
    const float * __restrict__ A,
    const float * __restrict__ B,
    float * __restrict__ C,
    int M, int K, int N
) {
    extern __shared__ char smem[];
    auto inst = GemmInstance((float*) smem, A, B, C, M, K, N);
    inst.run();
}


template <typename GemmInstance>
bool run_kernel(
    const float * __restrict__ A,
    const float * __restrict__ B,
    float * __restrict__ C,
    int M, int K, int N
) {
    hipLaunchKernelGGL(gemm_kernel<GemmInstance>, GemmInstance::blocks(M, K, N), GemmInstance::threads(), 0, 0, A, B, C, M, K, N);
    auto error = hipGetLastError();
    if (error != hipSuccess) {
        printf("Error: %s\n", hipGetErrorString(error));
        return false;
    }
    return true;
}

template <typename GemmInstance>
bool run_kernel_dyn_smem(
    const float * __restrict__ A,
    const float * __restrict__ B,
    float * __restrict__ C,
    int M, int K, int N
) {
    hipLaunchKernelGGL(gemm_kernel_dyn_smem<GemmInstance>, GemmInstance::blocks(M, K, N), GemmInstance::threads(), GemmInstance::used_smem_bytes(), 0, A, B, C, M, K, N);
    auto error = hipGetLastError();
    if (error != hipSuccess) {
        printf("Error: %s\n", hipGetErrorString(error));
        return false;
    }
    return true;
}

template <typename A>
__device__ void print_smem(A& a) {
    if (threadIdx.x == 0) {
        repeat<A::s1, A::s2>([&](int i, int j) {
            printf("%f ", (float) *a.index(i, j));
            if (j == A::s2 - 1)
                printf("\n");
        });
    }
}


