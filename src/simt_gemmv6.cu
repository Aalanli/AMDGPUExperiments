#include <cassert>
#include <stdio.h>
#include "utils.hpp"

#ifndef BlocksizeK_M
#define BlocksizeK_M 8
#endif

#ifndef BlocksizeK_N
#define BlocksizeK_N 4
#endif

#ifndef BlockWarpsM
#define BlockWarpsM 1
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
#define WarpMidM 4
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

constexpr int load_vector_factor(int n) {
    if (n % 4 == 0 && n >= 4) {
        return 4;
    } else if (n % 2 == 0 && n >= 2) {
        return 2;
    } else {
        return 1;
    }
}

constexpr int block_m = WarpOuterM * BlockWarpsM * WarpMidM * WarpInnerM;
constexpr int block_n = WarpOuterN * BlockWarpsN * WarpMidN * WarpInnerN;
constexpr int block_km = BlocksizeK_M;
constexpr int block_kn = BlocksizeK_N;
constexpr int nthreads = WarpMidM * WarpMidN * BlockWarpsM * BlockWarpsN;
constexpr int warp_size = WarpMidM * WarpMidN;

static_assert(nthreads % block_km == 0, "");
static_assert(nthreads % block_kn == 0, "");
static_assert(warp_size % block_km == 0, "");
static_assert(warp_size % block_kn == 0, "");
static_assert(block_km % 2 == 0, "");
static_assert(block_kn % 2 == 0, "");

constexpr int stride_ldg_m = nthreads / block_km;
constexpr int rep_ldg_m = block_m / stride_ldg_m;
constexpr int ldg_m_inner = load_vector_factor(rep_ldg_m);
constexpr int ldg_m_outer = rep_ldg_m / ldg_m_inner;
constexpr int ldg_rotate_m = (warp_size / block_km) * ldg_m_inner;

constexpr int stride_ldg_n = nthreads / block_kn;
constexpr int rep_ldg_n = block_n / stride_ldg_n;

static_assert(block_km % block_kn == 0 && block_km >= block_kn, "");

constexpr int used_smem = (block_m * block_km + block_n * block_kn) * sizeof(float);

__global__ __launch_bounds__(nthreads) void simt_gemm_kernelv6(
    const float* __restrict__ A, 
    const float* __restrict__ B, 
    float* __restrict__ C,
    int M, int N, int K
) {
    float regs_a[2][WarpOuterM][WarpInnerM];
    float regs_ldg_a[ldg_m_outer][ldg_m_inner];

    float regs_b[2][WarpOuterN][WarpInnerN];
    float regs_ldg_b[rep_ldg_n];

    float regs_c[WarpOuterM][WarpOuterN][WarpInnerM][WarpInnerN];

    __shared__ float sA[2][block_km][block_m];
    __shared__ float sB[2][block_kn][block_n];

    const int warp_idx = threadIdx.x / warp_size;
    const int warp_lane = threadIdx.x % warp_size;
    const int warp_m = warp_idx / BlockWarpsN;
    const int warp_n = warp_idx % BlockWarpsN;
    const int tm = warp_lane / WarpMidN;
    const int tn = warp_lane % WarpMidN;

    auto copy_a_g2r = [&](int offset_k) {
        int offset_m = blockIdx.x * block_m + (threadIdx.x / block_km) * ldg_m_inner;
        int coord_k = offset_k + threadIdx.x % block_km;
        for (int i = 0; i < ldg_m_outer; ++i) {
            for (int j = 0; j < ldg_m_inner; ++j) {
                int coord_m = offset_m + j + i * stride_ldg_m * ldg_m_inner;
                bool inbounds = coord_m < M && coord_k < K;
                regs_ldg_a[i][j] = inbounds ? A[coord_m * K + coord_k] : 0.0f;
            }
        }
    };

    auto copy_a_r2s = [&](int slice_idx) {
        int offset_m = (threadIdx.x / block_km) * ldg_m_inner;
        int offset_k = threadIdx.x % block_km;
        for (int i = 0; i < ldg_m_outer; ++i) {
            int coord_m = offset_m + i * stride_ldg_m * ldg_m_inner;
            int smem_offset = (coord_m + offset_k * ldg_rotate_m) % block_m;
            if (ldg_m_inner == 4) {
                float4* ptr = reinterpret_cast<float4*>(regs_ldg_a[i]);
                float4* smem = reinterpret_cast<float4*>(sA[slice_idx][offset_k]);
                smem[smem_offset / 4] = ptr[0];
            } else if (ldg_m_inner == 2) {
                float2* ptr = reinterpret_cast<float2*>(regs_ldg_a[i]);
                float2* smem = reinterpret_cast<float2*>(sA[slice_idx][offset_k]);
                smem[smem_offset / 2] = ptr[0];
            } else {
                float* ptr = reinterpret_cast<float*>(regs_ldg_a[i]);
                float* smem = reinterpret_cast<float*>(sA[slice_idx][offset_k]);
                smem[smem_offset] = ptr[0];
            }
            // for (int j = 0; j < ldg_m_inner; ++j) {
            //     sA[slice_idx][offset_k][(coord_m + offset_k * ldg_rotate_m) % block_m] = regs_ldg_a[i][j];
            //     // assert(coord_m < block_m&&"out of bounds sA");
            //     // assert(slice_idx < 2&&"out of bounds sA");
            //     // assert(offset_k < block_km&&"out of bounds sA");
            // }
        }
    };

    auto copy_a_s2r = [&](int slice_smem, int slice_regs, int k_frag_idx) {
        for (int wm = 0; wm < WarpOuterM; ++wm) {
            for (int wi = 0; wi < WarpInnerM; ++wi) {
                int i = wi + tm * WarpInnerM + wm * WarpMidM * WarpInnerM + warp_m * WarpOuterM * WarpMidM * WarpInnerM;
                // assert(slice_smem < 2&&"out of bounds sA");
                // assert(k_frag_idx < block_km&&"out of bounds sA");
                // assert(i < block_m&&"out of bounds sA");
                regs_a[slice_regs][wm][wi] = sA[slice_smem][k_frag_idx][(i + k_frag_idx * ldg_rotate_m) % block_m];
            }
        }
    };

    auto copy_b_g2r = [&](int offset_k) {
        int offset_n = (blockIdx.y * block_n + threadIdx.x % stride_ldg_n);
        int offset_k_ = offset_k + threadIdx.x / stride_ldg_n;
        for (int i = 0; i < rep_ldg_n; ++i) {
            bool inbounds = (offset_n + i * stride_ldg_n) < N && offset_k_ < K;
            regs_ldg_b[i] = inbounds ? B[offset_k_ * N + offset_n + i * stride_ldg_n] : 0.0f;
        }
    };

    auto copy_b_r2s = [&](int slice_idx) {
        int offset_n = threadIdx.x % stride_ldg_n;
        int offset_k = threadIdx.x / stride_ldg_n;
        for (int i = 0; i < rep_ldg_n; ++i) {
            // assert(slice_idx < 2&&"out of bounds sB");
            // assert(offset_k < block_kn&&"out of bounds sB");
            // assert(offset_n + i * stride_ldg_n < block_n&&"out of bounds sB");
            sB[slice_idx][offset_k][offset_n + i * stride_ldg_n] = regs_ldg_b[i];
        }
    };

    auto copy_b_s2r = [&](int slice_smem, int slice_regs, int k_frag_idx) {
        for (int wn = 0; wn < WarpOuterN; ++wn) {
            for (int wi = 0; wi < WarpInnerN; ++wi) {
                int i = wi + tn * WarpInnerN + wn * WarpMidN * WarpInnerN + warp_n * WarpOuterN * WarpMidN * WarpInnerN;
                // assert(slice_smem < 2&&"out of bounds sB");
                // assert(k_frag_idx < block_kn&&"out of bounds sB");
                // assert(i < block_n&&"out of bounds sB");
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

    
    copy_a_g2r(0); // offset k
    copy_a_r2s(0); // slice idx
    copy_b_g2r(0);  
    copy_b_r2s(0);
    __syncthreads();
    copy_a_s2r(0, 0, 0); // slice smem, slice regs, k frag idx
    copy_b_s2r(0, 0, 0);

    int k_tiles = (K + block_km - 1) / block_km - 1;
    for (int k = 0; k < k_tiles; ++k) {
        int offset_k = (k + 1) * block_km;
        copy_a_g2r(offset_k);
        #pragma unroll
        for (int k_frag = 0; k_frag < block_km; ++k_frag) {
            if (k_frag == block_km - 1)
                copy_a_r2s((k + 1) % 2); // smem slice idx
            if (k_frag % block_kn == block_kn - 1)
                copy_b_r2s((k * (block_km / block_kn) + k_frag / block_kn + 1) % 2); // smem slice idx
            __syncthreads();
            if (k_frag == block_km - 1)
                copy_a_s2r((k + 1) % 2, 0, 0); // smem slice, regs slice, k idx
            else
                copy_a_s2r(k % 2, (k_frag + 1) % 2, k_frag + 1); // smem slice, regs slice, k idx
            if (k_frag % block_kn == block_kn - 1)
                copy_b_s2r((k * (block_km / block_kn) + k_frag / block_kn + 1) % 2, 0, 0);
            else
                copy_b_s2r((k * (block_km / block_kn) + k_frag / block_kn) % 2, (k_frag + 1) % 2, (k_frag + 1) % block_kn);
            
            if (k_frag % block_kn == 0) {
                copy_b_g2r(k * block_km + (k_frag/block_kn + 1) * block_kn);
            }
            
            mma(k_frag % 2);
        }
    }

    int offset_k = k_tiles * block_km;
    copy_a_g2r(offset_k);
    #pragma unroll
    for (int k_frag = 0; k_frag < block_km; ++k_frag) {
        if (k_frag % block_kn == block_kn - 1 && k_frag != block_km - 1) {
            copy_b_r2s((k_tiles * (block_km / block_kn) + k_frag / block_kn + 1) % 2); // smem slice idx
            __syncthreads();
        }
        copy_a_s2r(k_tiles % 2, (k_frag + 1) % 2, k_frag + 1); // smem slice, regs slice, k idx
        if (k_frag % block_kn == block_kn - 1)
            copy_b_s2r((k_tiles * (block_km / block_kn) + k_frag / block_kn + 1) % 2, 0, 0);
        else
            copy_b_s2r((k_tiles * (block_km / block_kn) + k_frag / block_kn) % 2, (k_frag + 1) % 2, (k_frag + 1) % block_kn);
        
        if (k_frag % block_kn == 0) {
            copy_b_g2r(k_tiles * block_km + (k_frag/block_kn + 1) * block_kn);
        }

        mma(k_frag % 2);
    }

    copy_c_r2g();
}


#ifndef LAUNCH_NAME
#define LAUNCH_NAME simt_gemm_v6
#endif


extern "C" __attribute__((visibility("default"))) bool LAUNCH_NAME(float* a, float* b, float* c, int m, int k, int n) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    int warp_size = prop.warpSize;
    int smem = prop.sharedMemPerBlock;
    int regs = prop.regsPerBlock;

    auto cdiv = [](int a, int b) { return (a + b - 1) / b; };
    dim3 grid(cdiv(m, block_m), cdiv(n, block_n));
    dim3 block(WarpMidM * WarpMidN * BlockWarpsM * BlockWarpsN);

    simt_gemm_kernelv6<<<grid, block>>>(a, b, c, m, n, k);

    // check error
    auto error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("Error: %s\n", cudaGetErrorString(error));
        return false;
    }

    return true;
}


