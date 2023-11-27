#include <stdio.h>
#include <hip/amd_detail/amd_hip_runtime.h>
#include "hip_utils.hpp"

#ifndef BlockWarpsK
#define BlockWarpsK 4
#endif

#ifndef BlockWarpsM
#define BlockWarpsM 2
#endif

#ifndef BlockWarpsN
#define BlockWarpsN 2
#endif

#ifndef WarpOuterM
#define WarpOuterM 2
#endif

#ifndef WarpOuterN
#define WarpOuterN 2
#endif

#ifndef WarpMidM
#define WarpMidM 8
#endif

#ifndef WarpMidN
#define WarpMidN 8
#endif

#ifndef WarpInnerM
#define WarpInnerM 4
#endif

#ifndef WarpInnerN
#define WarpInnerN 4
#endif

constexpr int block_m = WarpOuterM * BlockWarpsM * WarpMidM * WarpInnerM;
constexpr int block_n = WarpOuterN * BlockWarpsN * WarpMidN * WarpInnerN;
constexpr int block_k = BlockWarpsK;
constexpr int nthreads = WarpMidM * WarpMidN * BlockWarpsM * BlockWarpsN;
// constexpr int warp_size = WarpMidM * WarpMidN;
static_assert(block_k % 2 == 0, "");
static_assert(block_k <= warp_size, "");
static_assert(warp_size % block_k == 0, "");
static_assert(block_m % (nthreads / block_k) == 0, "");
static_assert(block_n % (nthreads / block_k) == 0, "");

constexpr int used_smem = (block_m * block_k + block_n * block_k) * sizeof(float) * 2;


__global__ __launch_bounds__(nthreads) void simt_gemm_hidet_kernel(
    const float* __restrict__ A, 
    const float* __restrict__ B, 
    float* __restrict__ C,
    int M, int N, int K
) {
    constexpr int lines = nthreads / block_k;

    float regs_a[2][WarpOuterM][WarpInnerM];
    constexpr int ldg_a = block_m / lines;
    float regs_ldg_a[ldg_a];

    float regs_b[2][WarpOuterN][WarpInnerN];
    constexpr int ldg_b = block_n / lines;
    float regs_ldg_b[ldg_b];

    float regs_c[WarpOuterM][WarpOuterN][WarpInnerM][WarpInnerN];

    __shared__ float sA[2][block_k][block_m];
    __shared__ float sB[2][block_k][block_n];

    const int warp_idx = threadIdx.x / warp_size;
    const int warp_lane = threadIdx.x % warp_size;
    const int warp_m = warp_idx / BlockWarpsN;
    const int warp_n = warp_idx % BlockWarpsN;
    const int tm = warp_lane / WarpMidN;
    const int tn = warp_lane % WarpMidN;

    auto copy_a_g2r = [&](int offset_k) {
        int offset_m = blockIdx.x * block_m + (threadIdx.x / block_k) * ldg_a;
        int offset_k_ = offset_k + threadIdx.x % block_k;
        for (int i = 0; i < ldg_a; ++i) {
            bool inbounds = (offset_m + i) < M && offset_k_ < K;
            regs_ldg_a[i] = inbounds ? A[(offset_m + i) * K + offset_k_] : 0.0f;
        }
    };

    auto copy_a_r2s = [&](int slice_idx) {
        int offset_m = (threadIdx.x / block_k) * ldg_a;
        int offset_k = threadIdx.x % block_k;
        for (int i = 0; i < ldg_a; ++i) {
            sA[slice_idx][offset_k][offset_m + i] = regs_ldg_a[i];
        }
    };

    auto copy_a_s2r = [&](int slice_smem, int slice_regs, int k_frag_idx) {
        for (int wm = 0; wm < WarpOuterM; ++wm) {
            for (int wi = 0; wi < WarpInnerM; ++wi) {
                int i = wi + tm * WarpInnerM + wm * WarpMidM * WarpInnerM + warp_m * WarpOuterM * WarpMidM * WarpInnerM;
                regs_a[slice_regs][wm][wi] = sA[slice_smem][k_frag_idx][i];
            }
        }
    };

    auto copy_b_g2r = [&](int offset_k) {
        int offset_n = (blockIdx.y * block_n + threadIdx.x % lines);
        int offset_k_ = offset_k + threadIdx.x / lines;
        for (int i = 0; i < ldg_b; ++i) {
            bool inbounds = (offset_n + i * lines) < N && offset_k_ < K;
            regs_ldg_b[i] = inbounds ? B[offset_k_ * N + offset_n + i * lines] : 0.0f;
        }
    };

    auto copy_b_r2s = [&](int slice_idx) {
        int offset_n = threadIdx.x % lines;
        int offset_k = threadIdx.x / lines;
        for (int i = 0; i < ldg_b; ++i) {
            sB[slice_idx][offset_k][offset_n + i * lines] = regs_ldg_b[i];
        }
    };

    auto copy_b_s2r = [&](int slice_smem, int slice_regs, int k_frag_idx) {
        for (int wn = 0; wn < WarpOuterN; ++wn) {
            for (int wi = 0; wi < WarpInnerN; ++wi) {
                int i = wi + tn * WarpInnerN + wn * WarpMidN * WarpInnerN + warp_n * WarpOuterN * WarpMidN * WarpInnerN;
                regs_b[slice_regs][wn][wi] = sB[slice_smem][k_frag_idx][i];
            }
        }
    };

    auto copy_c_r2g = [&]() {
        for (int wm = 0; wm < WarpOuterM; ++wm) {
            for (int wn = 0; wn < WarpOuterN; ++wn) {
                for (int tim = 0; tim < WarpInnerM; tim++) {
                    for (int tin = 0; tin < WarpInnerN; tin++) {
                        int offset_m = tim + tm * WarpInnerM + wm * WarpMidM * WarpInnerM + warp_m * WarpOuterM * WarpMidM * WarpInnerM + blockIdx.x * block_m;
                        int offset_n = tin + tn * WarpInnerN + wn * WarpMidN * WarpInnerN + warp_n * WarpOuterN * WarpMidN * WarpInnerN + blockIdx.y * block_n;
                        bool inbounds = offset_m < M && offset_n < N;
                        if (inbounds) {
                            C[offset_m * N + offset_n] = regs_c[wm][wn][tim][tin];
                        }
                    }
                }
            }
        }
    };
    
    auto mma = [&](int idx) {
        for (int wm = 0; wm < WarpOuterM; ++wm)
            for (int wn = 0; wn < WarpOuterN; ++wn)
                for (int tim = 0; tim < WarpInnerM; tim++)
                    for (int tin = 0; tin < WarpInnerN; tin++)
                        regs_c[wm][wn][tim][tin] += regs_a[idx][wm][tim] * regs_b[idx][wn][tin];
    };

    for (int wm = 0; wm < WarpOuterM; ++wm)
        for (int wn = 0; wn < WarpOuterN; ++wn)
            for (int tim = 0; tim < WarpInnerM; tim++)
                for (int tin = 0; tin < WarpInnerN; tin++)
                    regs_c[wm][wn][tim][tin] = 0.0f;

    copy_a_g2r(0);
    copy_a_r2s(0);
    copy_b_g2r(0);
    copy_b_r2s(0);
    __syncthreads();
    copy_a_s2r(0, 0, 0);
    copy_b_s2r(0, 0, 0);

    int k_tiles = (K + block_k - 1) / block_k - 1;
    for (int k = 0; k < k_tiles; ++k) {
        int offset_k = (k + 1) * block_k;
        #pragma unroll
        for (int k_frag = 0; k_frag < BlockWarpsK; ++k_frag) {
            if (k_frag == BlockWarpsK - 1) {
                copy_a_r2s((k + 1) % 2);
                copy_b_r2s((k + 1) % 2);
                __syncthreads();
                copy_a_s2r((k + 1) % 2, (k_frag + 1) % 2, 0);
                copy_b_s2r((k + 1) % 2, (k_frag + 1) % 2, 0);
            } else {
                copy_a_s2r(k % 2, (k_frag + 1) % 2, k_frag + 1);
                copy_b_s2r(k % 2, (k_frag + 1) % 2, k_frag + 1);
            }
            if (k_frag == 0) {
                copy_a_g2r(offset_k);
                copy_b_g2r(offset_k);
            }
            mma(k_frag % 2);
        }
    }
    #pragma unroll
    for (int k_frag = 0; k_frag < BlockWarpsK; ++k_frag) {
        if (k_frag < BlockWarpsK - 1) {
            copy_a_s2r(k_tiles % 2, (k_frag + 1) % 2, k_frag + 1);
            copy_b_s2r(k_tiles % 2, (k_frag + 1) % 2, k_frag + 1);
        }
        mma(k_frag % 2);
    }

    copy_c_r2g();
}


__global__ __launch_bounds__(nthreads) void simt_gemm_hidet_kernelv2(
    const float* __restrict__ A, 
    const float* __restrict__ B, 
    float* __restrict__ C,
    int M, int N, int K
) {
    constexpr int lines = nthreads / block_k;

    float regs_a[2][WarpOuterM][WarpInnerM];
    constexpr int ldg_a = block_m / lines;
    float regs_ldg_a[ldg_a];

    float regs_b[2][WarpOuterN][WarpInnerN];
    constexpr int ldg_b = block_n / lines;
    float regs_ldg_b[ldg_b];

    float regs_c[WarpOuterM][WarpOuterN][WarpInnerM][WarpInnerN];

    __shared__ float sA[2][block_k][block_m];
    __shared__ float sB[2][block_k][block_n];

    const int warp_idx = threadIdx.x / warp_size;
    const int warp_lane = threadIdx.x % warp_size;
    const int warp_m = warp_idx / BlockWarpsN;
    const int warp_n = warp_idx % BlockWarpsN;
    const int tm = warp_lane / WarpMidN;
    const int tn = warp_lane % WarpMidN;

    auto copy_a_g2r = [&](int offset_k) {
        int offset_m = blockIdx.x * block_m + (threadIdx.x / block_k) * ldg_a;
        int offset_k_ = offset_k + threadIdx.x % block_k;
        for (int i = 0; i < ldg_a; ++i) {
            bool inbounds = (offset_m + i) < M && offset_k_ < K;
            regs_ldg_a[i] = inbounds ? A[(offset_m + i) * K + offset_k_] : 0.0f;
        }
    };

    auto copy_a_r2s = [&](int slice_idx) {
        int offset_m = (threadIdx.x / block_k) * ldg_a;
        int offset_k = threadIdx.x % block_k;
        for (int i = 0; i < ldg_a; ++i) {
            sA[slice_idx][offset_k][offset_m + i] = regs_ldg_a[i];
        }
    };

    auto copy_a_s2r = [&](int slice_smem, int slice_regs, int k_frag_idx) {
        for (int wm = 0; wm < WarpOuterM; ++wm) {
            for (int wi = 0; wi < WarpInnerM; ++wi) {
                int i = wi + tm * WarpInnerM + wm * WarpMidM * WarpInnerM + warp_m * WarpOuterM * WarpMidM * WarpInnerM;
                regs_a[slice_regs][wm][wi] = sA[slice_smem][k_frag_idx][i];
            }
        }
    };

    auto copy_b_g2r = [&](int offset_k) {
        int offset_n = (blockIdx.y * block_n + threadIdx.x % lines);
        int offset_k_ = offset_k + threadIdx.x / lines;
        for (int i = 0; i < ldg_b; ++i) {
            bool inbounds = (offset_n + i * lines) < N && offset_k_ < K;
            regs_ldg_b[i] = inbounds ? B[offset_k_ * N + offset_n + i * lines] : 0.0f;
        }
    };

    auto copy_b_r2s = [&](int slice_idx) {
        int offset_n = threadIdx.x % lines;
        int offset_k = threadIdx.x / lines;
        for (int i = 0; i < ldg_b; ++i) {
            sB[slice_idx][offset_k][offset_n + i * lines] = regs_ldg_b[i];
        }
    };

    auto copy_b_s2r = [&](int slice_smem, int slice_regs, int k_frag_idx) {
        for (int wn = 0; wn < WarpOuterN; ++wn) {
            for (int wi = 0; wi < WarpInnerN; ++wi) {
                int i = wi + tn * WarpInnerN + wn * WarpMidN * WarpInnerN + warp_n * WarpOuterN * WarpMidN * WarpInnerN;
                regs_b[slice_regs][wn][wi] = sB[slice_smem][k_frag_idx][i];
            }
        }
    };

    auto copy_c_r2g = [&]() {
        for (int wm = 0; wm < WarpOuterM; ++wm) {
            for (int wn = 0; wn < WarpOuterN; ++wn) {
                for (int tim = 0; tim < WarpInnerM; tim++) {
                    for (int tin = 0; tin < WarpInnerN; tin++) {
                        int offset_m = tim + tm * WarpInnerM + wm * WarpMidM * WarpInnerM + warp_m * WarpOuterM * WarpMidM * WarpInnerM + blockIdx.x * block_m;
                        int offset_n = tin + tn * WarpInnerN + wn * WarpMidN * WarpInnerN + warp_n * WarpOuterN * WarpMidN * WarpInnerN + blockIdx.y * block_n;
                        bool inbounds = offset_m < M && offset_n < N;
                        if (inbounds) {
                            C[offset_m * N + offset_n] = regs_c[wm][wn][tim][tin];
                        }
                    }
                }
            }
        }
    };
    
    auto mma = [&](int idx) {
        for (int wm = 0; wm < WarpOuterM; ++wm)
            for (int wn = 0; wn < WarpOuterN; ++wn)
                for (int tim = 0; tim < WarpInnerM; tim++)
                    for (int tin = 0; tin < WarpInnerN; tin++)
                        regs_c[wm][wn][tim][tin] += regs_a[idx][wm][tim] * regs_b[idx][wn][tin];
    };

    for (int wm = 0; wm < WarpOuterM; ++wm)
        for (int wn = 0; wn < WarpOuterN; ++wn)
            for (int tim = 0; tim < WarpInnerM; tim++)
                for (int tin = 0; tin < WarpInnerN; tin++)
                    regs_c[wm][wn][tim][tin] = 0.0f;

    copy_a_g2r(0);
    copy_a_r2s(0);
    copy_b_g2r(0);
    copy_b_r2s(0);
    __syncthreads();
    copy_a_s2r(0, 0, 0);
    copy_b_s2r(0, 0, 0);

    int k_tiles = (K + block_k - 1) / block_k - 1;
    for (int k = 0; k < k_tiles; ++k) {
        int offset_k = (k + 1) * block_k;
        copy_a_g2r(offset_k);
        copy_b_g2r(offset_k);
        #pragma unroll
        for (int k_frag = 0; k_frag < BlockWarpsK - 1; ++k_frag) {
            copy_a_s2r(k % 2, (k_frag + 1) % 2, k_frag + 1);
            copy_b_s2r(k % 2, (k_frag + 1) % 2, k_frag + 1);
            mma(k_frag % 2);
        }
        mma((BlockWarpsK - 1) % 2);
        copy_a_r2s((k + 1) % 2);
        copy_b_r2s((k + 1) % 2);
        __syncthreads();
        copy_a_s2r((k + 1) % 2, 0, 0);
        copy_b_s2r((k + 1) % 2, 0, 0);
    }
    #pragma unroll
    for (int k_frag = 0; k_frag < BlockWarpsK; ++k_frag) {
        if (k_frag < BlockWarpsK - 1) {
            copy_a_s2r(k_tiles % 2, (k_frag + 1) % 2, k_frag + 1);
            copy_b_s2r(k_tiles % 2, (k_frag + 1) % 2, k_frag + 1);
        }
        mma(k_frag % 2);
    }

    copy_c_r2g();
}

__global__ __launch_bounds__(nthreads) void simt_gemm_hidet_kernelv3(
    const float* __restrict__ A, 
    const float* __restrict__ B, 
    float* __restrict__ C,
    int M, int N, int K
) {
    constexpr int lines = nthreads / block_k;

    float regs_a[2][WarpOuterM][WarpInnerM];
    constexpr int ldg_a = block_m / lines;
    float regs_ldg_a[ldg_a];

    float regs_b[2][WarpOuterN][WarpInnerN];
    constexpr int ldg_b = block_n / lines;
    float regs_ldg_b[ldg_b];

    float regs_c[WarpOuterM][WarpOuterN][WarpInnerM][WarpInnerN];

    __shared__ float sA[2][block_k][block_m];
    __shared__ float sB[2][block_k][block_n];

    const int warp_idx = threadIdx.x / warp_size;
    const int warp_lane = threadIdx.x % warp_size;
    const int warp_m = warp_idx / BlockWarpsN;
    const int warp_n = warp_idx % BlockWarpsN;
    const int tm = warp_lane / WarpMidN;
    const int tn = warp_lane % WarpMidN;

    auto copy_a_g2r = [&](int offset_k) {
        int offset_m = blockIdx.x * block_m + (threadIdx.x / block_k) * ldg_a;
        int offset_k_ = offset_k + threadIdx.x % block_k;
        for (int i = 0; i < ldg_a; ++i) {
            bool inbounds = (offset_m + i) < M && offset_k_ < K;
            regs_ldg_a[i] = inbounds ? A[(offset_m + i) * K + offset_k_] : 0.0f;
        }
    };

    auto copy_a_r2s = [&](int slice_idx) {
        int offset_m = (threadIdx.x / block_k) * ldg_a;
        int offset_k = threadIdx.x % block_k;
        for (int i = 0; i < ldg_a; ++i) {
            sA[slice_idx][offset_k][offset_m + i] = regs_ldg_a[i];
        }
    };

    auto copy_a_s2r = [&](int slice_smem, int slice_regs, int k_frag_idx) {
        for (int wm = 0; wm < WarpOuterM; ++wm) {
            for (int wi = 0; wi < WarpInnerM; ++wi) {
                int i = wi + tm * WarpInnerM + wm * WarpMidM * WarpInnerM + warp_m * WarpOuterM * WarpMidM * WarpInnerM;
                regs_a[slice_regs][wm][wi] = sA[slice_smem][k_frag_idx][i];
            }
        }
    };

    auto copy_b_g2r = [&](int offset_k) {
        int offset_n = (blockIdx.y * block_n + threadIdx.x % lines);
        int offset_k_ = offset_k + threadIdx.x / lines;
        for (int i = 0; i < ldg_b; ++i) {
            bool inbounds = (offset_n + i * lines) < N && offset_k_ < K;
            regs_ldg_b[i] = inbounds ? B[offset_k_ * N + offset_n + i * lines] : 0.0f;
        }
    };

    auto copy_b_r2s = [&](int slice_idx) {
        int offset_n = threadIdx.x % lines;
        int offset_k = threadIdx.x / lines;
        for (int i = 0; i < ldg_b; ++i) {
            sB[slice_idx][offset_k][offset_n + i * lines] = regs_ldg_b[i];
        }
    };

    auto copy_b_s2r = [&](int slice_smem, int slice_regs, int k_frag_idx) {
        for (int wn = 0; wn < WarpOuterN; ++wn) {
            for (int wi = 0; wi < WarpInnerN; ++wi) {
                int i = wi + tn * WarpInnerN + wn * WarpMidN * WarpInnerN + warp_n * WarpOuterN * WarpMidN * WarpInnerN;
                regs_b[slice_regs][wn][wi] = sB[slice_smem][k_frag_idx][i];
            }
        }
    };

    auto copy_c_r2g = [&]() {
        for (int wm = 0; wm < WarpOuterM; ++wm) {
            for (int wn = 0; wn < WarpOuterN; ++wn) {
                for (int tim = 0; tim < WarpInnerM; tim++) {
                    for (int tin = 0; tin < WarpInnerN; tin++) {
                        int offset_m = tim + tm * WarpInnerM + wm * WarpMidM * WarpInnerM + warp_m * WarpOuterM * WarpMidM * WarpInnerM + blockIdx.x * block_m;
                        int offset_n = tin + tn * WarpInnerN + wn * WarpMidN * WarpInnerN + warp_n * WarpOuterN * WarpMidN * WarpInnerN + blockIdx.y * block_n;
                        bool inbounds = offset_m < M && offset_n < N;
                        if (inbounds) {
                            C[offset_m * N + offset_n] = regs_c[wm][wn][tim][tin];
                        }
                    }
                }
            }
        }
    };
    
    auto mma = [&](int idx) {
        for (int wm = 0; wm < WarpOuterM; ++wm)
            for (int wn = 0; wn < WarpOuterN; ++wn)
                for (int tim = 0; tim < WarpInnerM; tim++)
                    for (int tin = 0; tin < WarpInnerN; tin++)
                        regs_c[wm][wn][tim][tin] += regs_a[idx][wm][tim] * regs_b[idx][wn][tin];
    };

    for (int wm = 0; wm < WarpOuterM; ++wm)
        for (int wn = 0; wn < WarpOuterN; ++wn)
            for (int tim = 0; tim < WarpInnerM; tim++)
                for (int tin = 0; tin < WarpInnerN; tin++)
                    regs_c[wm][wn][tim][tin] = 0.0f;

    copy_a_g2r(0);
    copy_a_r2s(0);
    copy_b_g2r(0);
    copy_b_r2s(0);
    __syncthreads();
    copy_a_s2r(0, 0, 0);
    copy_b_s2r(0, 0, 0);

    int k_tiles = (K + block_k - 1) / block_k - 1;
    for (int k = 0; k < k_tiles; ++k) {
        int offset_k = (k + 1) * block_k;
        #pragma unroll
        for (int k_frag = 0; k_frag < BlockWarpsK - 1; ++k_frag) {
            mma(k_frag % 2);
            copy_a_s2r(k % 2, (k_frag + 1) % 2, k_frag + 1);
            copy_b_s2r(k % 2, (k_frag + 1) % 2, k_frag + 1);
        }
        mma((BlockWarpsK - 1) % 2);
        copy_a_g2r(offset_k);
        copy_b_g2r(offset_k);
        copy_a_r2s((k + 1) % 2);
        copy_b_r2s((k + 1) % 2);
        __syncthreads();
        copy_a_s2r((k + 1) % 2, 0, 0);
        copy_b_s2r((k + 1) % 2, 0, 0);
    }
    #pragma unroll
    for (int k_frag = 0; k_frag < BlockWarpsK; ++k_frag) {
        if (k_frag < BlockWarpsK - 1) {
            copy_a_s2r(k_tiles % 2, (k_frag + 1) % 2, k_frag + 1);
            copy_b_s2r(k_tiles % 2, (k_frag + 1) % 2, k_frag + 1);
        }
        mma(k_frag % 2);
    }

    copy_c_r2g();
}

__global__ __launch_bounds__(nthreads) void simt_gemm_hidet_kernelv4(
    const float* __restrict__ A, 
    const float* __restrict__ B, 
    float* __restrict__ C,
    int M, int N, int K
) {
    constexpr int lines = nthreads / block_k;

    float regs_a[1][WarpOuterM][WarpInnerM];
    constexpr int ldg_a = block_m / lines;
    float regs_ldg_a[ldg_a];

    float regs_b[1][WarpOuterN][WarpInnerN];
    constexpr int ldg_b = block_n / lines;
    float regs_ldg_b[ldg_b];

    float regs_c[WarpOuterM][WarpOuterN][WarpInnerM][WarpInnerN];

    __shared__ float sA[2][block_k][block_m];
    __shared__ float sB[2][block_k][block_n];

    const int warp_idx = threadIdx.x / warp_size;
    const int warp_lane = threadIdx.x % warp_size;
    const int warp_m = warp_idx / BlockWarpsN;
    const int warp_n = warp_idx % BlockWarpsN;
    const int tm = warp_lane / WarpMidN;
    const int tn = warp_lane % WarpMidN;

    auto copy_a_g2r = [&](int offset_k) {
        int offset_m = blockIdx.x * block_m + (threadIdx.x / block_k) * ldg_a;
        int offset_k_ = offset_k + threadIdx.x % block_k;
        for (int i = 0; i < ldg_a; ++i) {
            bool inbounds = (offset_m + i) < M && offset_k_ < K;
            regs_ldg_a[i] = inbounds ? A[(offset_m + i) * K + offset_k_] : 0.0f;
        }
    };

    auto copy_a_r2s = [&](int slice_idx) {
        int offset_m = (threadIdx.x / block_k) * ldg_a;
        int offset_k = threadIdx.x % block_k;
        for (int i = 0; i < ldg_a; ++i) {
            sA[slice_idx][offset_k][offset_m + i] = regs_ldg_a[i];
        }
    };

    auto copy_a_s2r = [&](int slice_smem, int slice_regs, int k_frag_idx) {
        for (int wm = 0; wm < WarpOuterM; ++wm) {
            for (int wi = 0; wi < WarpInnerM; ++wi) {
                int i = wi + tm * WarpInnerM + wm * WarpMidM * WarpInnerM + warp_m * WarpOuterM * WarpMidM * WarpInnerM;
                regs_a[slice_regs][wm][wi] = sA[slice_smem][k_frag_idx][i];
            }
        }
    };

    auto copy_b_g2r = [&](int offset_k) {
        int offset_n = (blockIdx.y * block_n + threadIdx.x % lines);
        int offset_k_ = offset_k + threadIdx.x / lines;
        for (int i = 0; i < ldg_b; ++i) {
            bool inbounds = (offset_n + i * lines) < N && offset_k_ < K;
            regs_ldg_b[i] = inbounds ? B[offset_k_ * N + offset_n + i * lines] : 0.0f;
        }
    };

    auto copy_b_r2s = [&](int slice_idx) {
        int offset_n = threadIdx.x % lines;
        int offset_k = threadIdx.x / lines;
        for (int i = 0; i < ldg_b; ++i) {
            sB[slice_idx][offset_k][offset_n + i * lines] = regs_ldg_b[i];
        }
    };

    auto copy_b_s2r = [&](int slice_smem, int slice_regs, int k_frag_idx) {
        for (int wn = 0; wn < WarpOuterN; ++wn) {
            for (int wi = 0; wi < WarpInnerN; ++wi) {
                int i = wi + tn * WarpInnerN + wn * WarpMidN * WarpInnerN + warp_n * WarpOuterN * WarpMidN * WarpInnerN;
                regs_b[slice_regs][wn][wi] = sB[slice_smem][k_frag_idx][i];
            }
        }
    };

    auto copy_c_r2g = [&]() {
        for (int wm = 0; wm < WarpOuterM; ++wm) {
            for (int wn = 0; wn < WarpOuterN; ++wn) {
                for (int tim = 0; tim < WarpInnerM; tim++) {
                    for (int tin = 0; tin < WarpInnerN; tin++) {
                        int offset_m = tim + tm * WarpInnerM + wm * WarpMidM * WarpInnerM + warp_m * WarpOuterM * WarpMidM * WarpInnerM + blockIdx.x * block_m;
                        int offset_n = tin + tn * WarpInnerN + wn * WarpMidN * WarpInnerN + warp_n * WarpOuterN * WarpMidN * WarpInnerN + blockIdx.y * block_n;
                        bool inbounds = offset_m < M && offset_n < N;
                        if (inbounds) {
                            C[offset_m * N + offset_n] = regs_c[wm][wn][tim][tin];
                        }
                    }
                }
            }
        }
    };
    
    auto mma = [&](int idx) {
        for (int wm = 0; wm < WarpOuterM; ++wm)
            for (int wn = 0; wn < WarpOuterN; ++wn)
                for (int tim = 0; tim < WarpInnerM; tim++)
                    for (int tin = 0; tin < WarpInnerN; tin++)
                        regs_c[wm][wn][tim][tin] += regs_a[idx][wm][tim] * regs_b[idx][wn][tin];
    };

    for (int wm = 0; wm < WarpOuterM; ++wm)
        for (int wn = 0; wn < WarpOuterN; ++wn)
            for (int tim = 0; tim < WarpInnerM; tim++)
                for (int tin = 0; tin < WarpInnerN; tin++)
                    regs_c[wm][wn][tim][tin] = 0.0f;

    copy_a_g2r(0);
    copy_a_r2s(0);
    copy_b_g2r(0);
    copy_b_r2s(0);
    __syncthreads();
    
    int k_tiles = (K + block_k - 1) / block_k - 1;
    for (int k = 0; k < k_tiles; ++k) {
        #pragma unroll
        for (int k_frag = 0; k_frag < BlockWarpsK; ++k_frag) {
            copy_a_s2r(k % 2, 0, k_frag);
            copy_b_s2r(k % 2, 0, k_frag);
            mma(0);
        }
        int offset_k = (k + 1) * block_k;
        copy_a_g2r(offset_k);
        copy_b_g2r(offset_k);
        copy_a_r2s((k + 1) % 2);
        copy_b_r2s((k + 1) % 2);
        __syncthreads();
    }
    #pragma unroll
    for (int k_frag = 0; k_frag < BlockWarpsK; ++k_frag) {
        copy_a_s2r(k_tiles % 2, 0, k_frag);
        copy_b_s2r(k_tiles % 2, 0, k_frag);
        mma(k_frag % 2);
    }

    copy_c_r2g();
}

#ifndef LAUNCH_NAME
#define LAUNCH_NAME simt_gemm_hidet
#endif


EXPORT bool LAUNCH_NAME(float* a, float* b, float* c, int m, int k, int n, int version) {
    hipDeviceProp_t prop;
    hipGetDeviceProperties(&prop, 0);
    int warp_size = prop.warpSize;
    int smem = prop.sharedMemPerBlock;
    int regs = prop.regsPerBlock;

    auto cdiv = [](int a, int b) { return (a + b - 1) / b; };
    dim3 grid(cdiv(m, block_m), cdiv(n, block_n));
    dim3 block(WarpMidM * WarpMidN * BlockWarpsM * BlockWarpsN);

    if (version == 0)
        hipLaunchKernelGGL(simt_gemm_hidet_kernel, grid, block, 0, 0, a, b, c, m, n, k);
    else if (version == 1)
        hipLaunchKernelGGL(simt_gemm_hidet_kernelv2, grid, block, 0, 0, a, b, c, m, n, k);
    else if (version == 2)
        hipLaunchKernelGGL(simt_gemm_hidet_kernelv3, grid, block, 0, 0, a, b, c, m, n, k);
    else if (version == 3)
        hipLaunchKernelGGL(simt_gemm_hidet_kernelv4, grid, block, 0, 0, a, b, c, m, n, k);
    else
        return false;

    // check error
    auto error = hipGetLastError();
    if (error != hipSuccess) {
        printf("Error: %s\n", hipGetErrorString(error));
        return false;
    }

    return true;
}
