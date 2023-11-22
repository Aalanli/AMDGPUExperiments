#include "hip_utils.hpp"
#include "mfma_tools.cpp"
#include <hip/amd_detail/amd_hip_runtime.h>
#include <stdio.h>


/// block level
template <typename T>
struct GMemTile {
    T* ptr;
    const int a;
    const int b; // hopefully the compiler can detect some of these are identical for a and b tiles (cse)
    int offset_a;
    int offset_b;

    __device__ T* index(int i, int j) {
        int coord_a = i + offset_a;
        int coord_b = i + offset_b;
        if (coord_a < a && coord_b < b) {
            return ptr + (coord_a * b + coord_b);
        } else {
            return nullptr;
        }
    }

    __device__ void inc_b(int i) { offset_b += i; }
    __device__ void inc_a(int i) { offset_a += i; }
};


/// block level
template <int BLOCK_A, int BLOCK_B, int Warps, int VecLoad>
struct LdgBlockFrag {
    static_assert(is_power_of_2(BLOCK_B) && BLOCK_B % VecLoad == 0, "");
    static_assert(is_power_of_2(BLOCK_A), "");
    static_assert(BLOCK_B <= Warps * warp_size, "not enough threads per row");
    static constexpr int threads_per_row = BLOCK_B / VecLoad;
    static constexpr int stride_col = Warps * warp_size / threads_per_row;
    static constexpr int rep_m = BLOCK_A / stride_col;
    static constexpr bool oversub = rep_m > BLOCK_A;
    
    float ldg_regs[rep_m][VecLoad];

    template <typename GmemAcc>
    __device__ void copy_g2r(GmemAcc& gA) {
        int row = threadIdx.x % (BLOCK_B * VecLoad);
        int col = threadIdx.x / (BLOCK_B * VecLoad);
        if constexpr(oversub) {
            if (col < BLOCK_A) {
                for (int i = 0; i < rep_m; ++i) {
                    vec_load_nullable<VecLoad>(gA.index(i * stride_col + col, row * VecLoad), &ldg_regs[i][0]);
                }    
            }
        } else {
            for (int i = 0; i < rep_m; ++i) {
                vec_load_nullable<VecLoad>(gA.index(i * stride_col + col, row * VecLoad), &ldg_regs[i][0]);
            }
        }
    }

    template <typename SmemAcc>
    __device__ void copy_r2s(SmemAcc& sA) {
        int row = threadIdx.x % (BLOCK_B * VecLoad);
        int col = threadIdx.x / (BLOCK_B * VecLoad);
        if constexpr(oversub) {
            if (col < BLOCK_A) {
                for (int i = 0; i < rep_m; ++i)
                    for (int j = 0; j < VecLoad; ++j)
                        *sA.index(i * stride_col + col, row * VecLoad + j) = ldg_regs[i][j];
            }
        } else {
            for (int i = 0; i < rep_m; ++i)
                // assume that the compiler can detect consecutive memory accesses and vectorize them
                for (int j = 0; j < VecLoad; ++j)
                    *sA.index(i * stride_col + col, row * VecLoad + j) = ldg_regs[i][j];            
        }
    }
};


/// warp level
template <int BLOCK_K>
struct MFMAF32_16x16F32_ATile {
    static constexpr int mma_m = 16;
    static constexpr int mma_k = 4;
    static constexpr int rep_k = BLOCK_K / mma_k;
    static_assert(is_power_of_2(BLOCK_K) && BLOCK_K > mma_k);

    float regs[rep_k];
    template <typename SmemAcc>
    __device__ void copy_s2r(SmemAcc& sA, int offset_m, int offset_k) {
        int lane = threadIdx.x % warp_size;
        if constexpr(rep_k % 4 == 0) {
            for (int i = 0; i < rep_k / 4; ++i) {
                for (int j = 0; j < 4; ++j) {
                    regs[i * 4 + j] = *sA.index(offset_m + (lane % mma_m), offset_k + (lane / mma_m + i * mma_k) * 4 + j);
                }
            }
        } else {
            for (int i = 0; i < rep_k; ++i) {
                regs[i] = *sA.index(offset_m + (lane % mma_m), offset_k + (lane / mma_m) + i * mma_k);
            }
        }
    }
};


/// warp level
template <int BLOCK_K>
struct MFMAF32_16x16F32_BTile {
    static constexpr int mma_k = 4;
    static constexpr int mma_n = 16;
    static constexpr int rep_k = BLOCK_K / mma_k;

    float regs[rep_k];
    template <typename SmemAcc>
    __device__ void copy_s2r(SmemAcc& sB, int offset_k, int offset_n) {
        int lane = threadIdx.x % warp_size;
        if constexpr(rep_k % 4 == 0) {
            for (int i = 0; i < rep_k / 4; ++i) {
                for (int j = 0; j < 4; ++j) {
                    regs[i * 4 + j] = *sB.index(offset_k + (lane / mma_n + i * mma_k) * 4 + j, offset_n + lane % mma_n);
                }
            }
        } else {
            for (int i = 0; i < rep_k; ++i) {
                regs[i] = *sB.index(offset_k + (lane / mma_n) + i * mma_k, offset_n + lane % mma_n);
            }
        }
    }
};


/// warp level
struct MFMAF32_16x16F32_CTile {
    using float4 = __attribute__( (__vector_size__(4 * sizeof(float)) )) float;
    float4 regs;

    __device__ void fill(float v) {
        regs[0] = v;
        regs[1] = v;
        regs[2] = v;
        regs[3] = v;
    }
};


/// warp level
template <int BLOCK_K>
void mma(MFMAF32_16x16F32_ATile<BLOCK_K> &atile, MFMAF32_16x16F32_BTile<BLOCK_K> &btile, MFMAF32_16x16F32_CTile &ctile) {
    constexpr int rep_k = MFMAF32_16x16F32_ATile<BLOCK_K>::rep_k;
    for (int i = 0; i < rep_k; ++i) {
        ctile.regs = __builtin_amdgcn_mfma_f32_16x16x4f32(
            atile.regs[i], btile.regs[i], ctile.regs, 0, 0, 0
        );
    }
}

template <int BLOCK_M, int BLOCK_K, int BLOCK_N, int Warps>
struct GemmMfma_f32_16x16x4_f32_Base : BasicGemmInstance<BLOCK_M, BLOCK_K, BLOCK_N, Warps> {
    using BasicGemmInstance<BLOCK_M, BLOCK_K, BLOCK_N, Warps>::BasicGemmInstance;
    static constexpr int mma_m = 16;
    static constexpr int mma_n = 16;
    static constexpr int mma_k = 4;
    static_assert(BLOCK_M % mma_m == 0, "");
    static_assert(BLOCK_N % mma_n == 0, "");
    static constexpr int warps_m = Min<BLOCK_M / mma_m, Warps>::value;
    static_assert(Warps % warps_m == 0, "");
    static constexpr int warps_n = Min<BLOCK_N / mma_n, Warps / warps_m>::value;

    static constexpr int rep_m = Max<BLOCK_M / (warps_m * mma_m), 1>::value;
    static constexpr int rep_n = Max<BLOCK_N / (warps_n * mma_n), 1>::value;
    // this could be removed if we check for oversubscription, eg: return from start
    static_assert(warps_m * warps_n == Warps, "");

    static constexpr int used_smem_bytes() {
        return (BLOCK_M + BLOCK_N) * BLOCK_K * sizeof(float);
    }

};