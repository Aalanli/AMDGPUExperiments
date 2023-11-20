#include <cstdio>
#include <hip/amd_detail/amd_hip_runtime.h>
#include <rocwmma/internal/types.hpp>
#include <rocwmma/rocwmma.hpp>

// 16, 32
#ifndef MMA_M
#define MMA_M 16
#endif
// 16, 32
#ifndef MMA_N
#define MMA_N 16
#endif

// 4, 8, 16, 32, 64, 128
#ifndef MMA_K
#define MMA_K 8
#endif

#ifndef REP_M
#define REP_M 1
#endif

#ifndef REP_N
#define REP_N 1
#endif

#ifndef WARP_M
#define WARP_M 1
#endif

#ifndef WARP_N
#define WARP_N 1
#endif

constexpr int warp_size = rocwmma::Constants::AMDGCN_WAVE_SIZE;
constexpr int nthreads = WARP_M * WARP_N * 64;
constexpr int block_m = MMA_M * REP_M * WARP_M;
constexpr int block_n = MMA_N * REP_N * WARP_N;
constexpr int block_k = MMA_K;

__global__ __launch_bounds__(nthreads) void gemm_wmma_f32_kernel(
    const float* __restrict__ a,
    const float* __restrict__ b,
    float* __restrict__ c,
    int m, int k, int n
) {
    using FragA = rocwmma::fragment<rocwmma::matrix_a, MMA_M, MMA_N, MMA_K, float, rocwmma::row_major>;
    using FragB = rocwmma::fragment<rocwmma::matrix_b, MMA_M, MMA_N, MMA_K, float, rocwmma::row_major>;
    using FragAcc = rocwmma::fragment<rocwmma::accumulator, MMA_M, MMA_N, MMA_K, float>;
    FragA atile[REP_M];
    FragB btile[REP_N];
    FragAcc ctile[REP_M][REP_N];

    const int warp_id = threadIdx.x / warp_size;
    const int warp_m = warp_id / WARP_N;
    const int warp_n = warp_id % WARP_N;
    const int offset_m = blockIdx.x * block_m + warp_m * REP_M * MMA_M;
    const int offset_n = blockIdx.y * block_n + warp_n * REP_N * MMA_N;

    for (int i = 0; i < REP_M; ++i) {
        for (int j = 0; j < REP_N; ++j) {
            rocwmma::fill_fragment(ctile[i][j], 0.0f);
        }
    }

    for (int kb = 0; kb < (k + block_k - 1) / block_k; ++kb) {
        #pragma unroll
        for (int mt = 0; mt < REP_M; mt++) {
            rocwmma::load_matrix_sync(atile[mt], 
                a + (offset_m + mt * MMA_M) * k + kb * block_k, k);
        }
        #pragma unroll
        for (int nt = 0; nt < REP_N; nt++) {
            rocwmma::load_matrix_sync(btile[nt],
                b + (kb * block_k) * n + (nt * MMA_N + offset_n), n);
        }

        for (int i = 0; i < REP_M; ++i) {
            for (int j = 0; j < REP_N; ++j) {
                rocwmma::mma_sync(ctile[i][j], atile[i], btile[j], ctile[i][j]);
            }
        }
    }

    for (int i = 0; i < REP_M; ++i) {
        for (int j = 0; j < REP_N; ++j) {
            rocwmma::store_matrix_sync(c + (offset_m + i * MMA_M) * n + (offset_n + j * MMA_N),
                ctile[i][j], n, rocwmma::mem_row_major);
        }
    }
}

#ifndef LAUNCH_NAME
#define LAUNCH_NAME wmma_gemm
#endif

extern "C" __attribute__((visibility("default"))) bool LAUNCH_NAME(float* a, float* b, float* c, int m, int k, int n) {
    // pretty sure masking is not supported yet
    if (m % block_m != 0 || n % block_n != 0 || k % block_k != 0) {
        printf("m,k,n not divisible by blocks{m,k,n}");
        return false;
    }
    dim3 block(nthreads);
    dim3 grid(m / block_m, n / block_n);
    hipLaunchKernelGGL(gemm_wmma_f32_kernel, grid, block, 0, 0, a, b, c, m, k, n);
    // check error
    auto error = hipGetLastError();
    if (error != hipSuccess) {
        printf("Error: %s\n", hipGetErrorString(error));
        return false;
    }

    return true;
}