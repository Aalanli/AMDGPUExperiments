#include "hip_utils.hpp"

#ifndef BLOCK_M
#define BLOCK_M 16
#endif

#ifndef BLOCK_N
#define BLOCK_N 16
#endif

#ifndef BLOCK_K
#define BLOCK_K 16
#endif

#ifndef Warp_M
#define Warp_M 2
#endif

#ifndef Warp_N
#define Warp_N 2
#endif


__global__ void mfma_f32_16x16x4f32_gemm_kernel(
    const float * __restrict__ A,
    const float * __restrict__ B,
    float * __restrict__ C,
    int M, int K, int N
) {
    using float4 = __attribute__( (__vector_size__(4 * sizeof(float)) )) float;
    constexpr int mma_m = 16;
    constexpr int mma_n = 16;
    constexpr int mma_k = 4;

    constexpr int rep_m = BLOCK_M / mma_m;
    constexpr int rep_n = BLOCK_N / mma_n;

    constexpr int nthreads = Warp_M * Warp_N * 64;

    float regs_a[rep_m];
    float regs_b[rep_n];
    float4 regs_c[rep_m][rep_n];

    auto mma = [&]() {
        for (int i = 0; i < rep_m; ++i) {
            for (int j = 0; j < rep_n; ++j) {
                regs_c[i][j] += __builtin_amdgcn_mfma_f32_16x16x4f32(
                    regs_a[i], regs_b[j], regs_c[i][j], 0, 0, 0);
            }
        }
    };

    const int warp_id = threadIdx.x / 64;
    const int lane = threadIdx.x % 64;
    const int warp_m = warp_id / Warp_N;
    const int warp_n = warp_id % Warp_N;

    __shared__ float sA[BLOCK_M][BLOCK_K];
    __shared__ float sB[BLOCK_K][BLOCK_N];
    
    static_assert(BLOCK_K <= nthreads && nthreads % BLOCK_K == 0, "");
    auto load_a_g2s = [&](int offset_k) {
        int row = threadIdx.x % BLOCK_K;
        int col = threadIdx.x / BLOCK_K;
        constexpr int stride_m = nthreads / BLOCK_K;
        constexpr int reps = BLOCK_M / stride_m;

        for (int i = 0; i < reps; ++i) {
            int coord_m = col + i * stride_m + blockIdx.x * BLOCK_M;
            int coord_k = row + offset_k;
            bool inbounds = coord_m < M && coord_k < K;
            sA[col + i * stride_m][row] = inbounds ? A[coord_m * K + coord_k] : 0;
        }
    };

    static_assert(BLOCK_N <= nthreads && nthreads % BLOCK_N == 0, "");
    auto load_b_g2s = [&](int offset_k) {
        int row = threadIdx.x % BLOCK_N;
        int col = threadIdx.x / BLOCK_N;
        constexpr int stride_k = nthreads / BLOCK_N;
        constexpr int reps = BLOCK_K / stride_k;

        for (int i = 0; i < reps; ++i) {
            int coord_k = col + i * stride_k + offset_k;
            int coord_n = row + blockIdx.y * BLOCK_N;
            bool inbounds = coord_k < K && coord_n < N;
            sB[col + i * stride_k][row] = inbounds ? B[coord_k * N + coord_n] : 0;
        }
    };

    auto load_a_s2r = [&](int k_idx) {
        int offset_m = lane % mma_m + warp_m * rep_m * mma_m;
        int offset_K = k_idx * mma_k + lane / mma_m;

        for (int i = 0; i < rep_m; ++i) {
            regs_a[i] = sA[offset_m + i * mma_m][offset_K];
        }
    };

    auto load_b_s2r = [&](int k_idx) {
        int offset_n = lane % mma_n + warp_n * rep_n * mma_n;
        int offset_k = k_idx * mma_k + lane / mma_n;

        for (int i = 0; i < rep_n; ++i) {
            regs_b[i] = sB[offset_k][offset_n + i * mma_n];
        }
    };

    auto store_c_r2g = [&]() {
        for (int im = 0; im < rep_m; ++im) {
            for (int in = 0; in < rep_n; ++in) {
                for (int k = 0; k < 4; ++k) {
                    int coord_m = k + (lane / 16) * 4 + im * mma_m + warp_m * rep_m * mma_m + blockIdx.x * BLOCK_M;
                    int coord_n = (lane % 16) + in * mma_n + warp_n * rep_n * mma_n + blockIdx.y * BLOCK_N;
                    bool inbounds = coord_m < M && coord_n < N;
                    if (inbounds) {
                        C[coord_m * N + coord_n] = regs_c[im][in][k];
                    }
                }
            }
        }
    };

    for (int im = 0; im < rep_m; im++) {
        for (int in = 0; in < rep_n; in++) {
            for (int k = 0; k < 4; ++k) {
                regs_c[im][in][k] = 0.0f;
            }
        }
    }

    for (int k = 0; k < cdiv(K, BLOCK_K); ++k) {
        load_a_g2s(k * BLOCK_K);
        load_b_g2s(k * BLOCK_K);
        __syncthreads();

        for (int i = 0; i < BLOCK_K / mma_k; ++i) {
            load_a_s2r(i);
            load_b_s2r(i);
            mma();
        }
        __syncthreads();
    }

    store_c_r2g();
}