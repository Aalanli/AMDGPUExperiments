#include "hip_utils.hpp"
#include <hip/amd_detail/amd_hip_runtime.h>


static constexpr int warp_size = 64;

#define DevHost __device__ __host__

template <int BLOCK_M, int BLOCK_K, int BLOCK_N, int Warps>
struct BasicGemmInstance {
    float* smem;
    const float* __restrict__ a;
    const float* __restrict__ b;
    float* __restrict__ c;
    const int m, n, k;
    static constexpr int used_smem_bytes() {
        return (BLOCK_M + BLOCK_N) * BLOCK_K * sizeof(float);
    }

    static constexpr int nthreads() {
        return Warps * warp_size;
    }

    DevHost BasicGemmInstance(
        float* smem, const float* a, const float* b, float* c, int m, int n, int k
    ) : smem(smem), a(a), b(b), c(c), m(m), n(n), k(k) {}

    static DevHost dim3 blocks(int m, int k, int n) {
        return dim3((m + BLOCK_M - 1) / BLOCK_M, (n + BLOCK_N - 1) / BLOCK_N);
    }

    static DevHost dim3 threads() {
        return dim3(nthreads());
    }
    
    __device__ void run() {
        assert(false && "NotImplemented");
    }
};

template <int A, int B>
struct Max {
    static constexpr int value = A < B ? B : A;
};

template <int A, int B>
struct Min {
    static constexpr int value = A < B ? A : B;
};


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
