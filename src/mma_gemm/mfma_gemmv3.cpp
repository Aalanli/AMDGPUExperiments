#include "hip_utils.hpp"
#include "mfma_tools.cpp"
#include <hip/amd_detail/amd_hip_runtime.h>
#include <hip/amd_detail/amd_warp_functions.h>
#include <stdio.h>
#include <tuple>


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
    static constexpr int tile_m = 16;
    static constexpr int tile_k = BLOCK_K;

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
    static constexpr int tile_n = 16;
    static constexpr int tile_k = BLOCK_K;

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

    template <typename GMemC>
    __device__ void copy_r2g(GMemC &gC, int offset_m, int offset_n) {
        int lane = threadIdx.x / warp_size;
        for (int i = 0; i < 4; ++i) {
            auto ptr = gC.index(offset_m + (lane / 16) * 4 + i, offset_n + lane % 16);
            if (ptr)
                *ptr = regs[i];
        }
    }
};


/// warp level

template <typename TileA, typename TileB, typename TileC>
struct TileMMA {
    __device__ static void mma(TileA& a, TileB& b, TileC& c) {}
};

template <int BLOCK_K>
struct TileMMA<MFMAF32_16x16F32_ATile<BLOCK_K>, MFMAF32_16x16F32_BTile<BLOCK_K>, MFMAF32_16x16F32_CTile> {
    __device__ static void mma(MFMAF32_16x16F32_ATile<BLOCK_K> &atile, MFMAF32_16x16F32_BTile<BLOCK_K> &btile, MFMAF32_16x16F32_CTile &ctile) {
        constexpr int rep_k = MFMAF32_16x16F32_ATile<BLOCK_K>::rep_k;
        for (int i = 0; i < rep_k; ++i) {
            ctile.regs = __builtin_amdgcn_mfma_f32_16x16x4f32(
                atile.regs[i], btile.regs[i], ctile.regs, 0, 0, 0
            );
        }
    }

};


/// block level
template <int BLOCK_M, 
          int BLOCK_K, 
          int BLOCK_N, 
          typename ATile, 
          typename BTile, 
          typename CTile, 
          int Warps>
struct GemmBlockT {
    static constexpr int tile_m = ATile::tile_m;
    static constexpr int tile_k = ATile::tile_k;
    static_assert(ATile::tile_k == BTile::tile_k, "");    
    static constexpr int tile_n = BTile::tile_n;

    static constexpr int block_m = BLOCK_M;
    static constexpr int block_k = BLOCK_K;
    static constexpr int block_n = BLOCK_N;
    static constexpr int warps = Warps;
    static_assert(block_m % tile_m == 0, "");
    static_assert(block_k % tile_k == 0, "");
    static_assert(block_n % tile_n == 0, "");

    static constexpr int warps_m = Min<BLOCK_M / tile_m, Warps>::value;
    static constexpr int warps_n = Min<BLOCK_N / tile_n, Warps / warps_m>::value;

    static constexpr int rep_m = Max<BLOCK_M / (warps_m * tile_m), 1>::value;
    static constexpr int rep_n = Max<BLOCK_N / (warps_n * tile_n), 1>::value;
    static constexpr int rep_k = BLOCK_K / tile_k;
    // this could be removed if we check for oversubscription, eg: return from start
    static_assert(warps_m * warps_n == Warps, "");

    CTile Ctiles[rep_m][rep_n];

    __device__ void fill_c(float v) {
        repeat<rep_m, rep_n>([&](int i, int j) {
            Ctiles[i][j].fill(v);
        }); 
    }

    template <typename GMemC>
    __device__ void copy_r2g(GMemC &gC) {
        int warp = threadIdx.x / warp_size;
        int warp_m = warp % warps_m;
        int warp_n = warp / warps_m;
        repeat<rep_m, rep_n>([&](int i, int j) {
            Ctiles[i][j].copy_g2r(&gC, warp_m * rep_m * tile_m + i * tile_m, warp_n * rep_n * tile_n + j * tile_n);
        });
    }

};

template <int BLOCK_M, 
          int BLOCK_K, 
          int BLOCK_N, 
          typename ATile, 
          typename BTile, 
          typename CTile, 
          typename MMA,
          int Warps>
struct GemmBlockT_V1 : GemmBlockT<BLOCK_M, BLOCK_K, BLOCK_N, ATile, BTile, CTile, Warps> {
    using super = GemmBlockT<BLOCK_M, BLOCK_K, BLOCK_N, ATile, BTile, CTile, Warps>;

    ATile Atiles[super::rep_m];
    BTile Btiles[super::rep_n];

    template <typename SMemA, typename SMemB>
    __device__ void mma(SMemA &sA, SMemB &sB) {
        int warp = threadIdx.x / warp_size;
        int warp_m = warp % super::warps_m;
        int warp_n = warp / super::warps_n;

        for (int k = 0; k < super::rep_k; k++) {
            // load m
            for (int im = 0; im < super::rep_m; im++) {
                Atiles[im].copy_s2r(
                    sA, 
                    warp_m * super::tile_m * super::rep_m + im * super::tile_m, 
                    k * super::tile_k
                );
            }

            // load n
            for (int in = 0; in < super::rep_n; in++) {
                Btiles[in].copy_s2r(
                    sB, 
                    k * super::tile_k,
                    warp_n * super::tile_n * super::rep_n + in * super::tile_n
                );
            }

            // mma
            repeat<super::rep_m, super::rep_n>([&](int i, int j) {
                MMA::mma(Atiles[i], Btiles[j], this->Ctiles[i][j]);
            });
        }
    }
};

template <int BLOCK_M, int BLOCK_K, int BLOCK_N, int VecLoad, int InnerK, int Warps>
KERNEL(Warps * warp_size) gemm_mfma_f32_16x16x4f32_v3(
    const float * __restrict__ A,
    const float * __restrict__ B,
    float * __restrict__ C,
    int M, int K, int N
) {
    const int offset_m = blockIdx.x * BLOCK_M;
    const int offset_n = blockIdx.y * BLOCK_N;
    GMemTile<const float> gA = {A, M, K, offset_m, 0};
    GMemTile<const float> gB = {B, K, N, 0, offset_n};

    __shared__ float smem[(BLOCK_N + BLOCK_M) * BLOCK_K];

    SMemSwizzleLayout<BLOCK_M, BLOCK_K, VecLoad> sA = {smem};
    SMemSwizzleLayout<BLOCK_K, BLOCK_N, VecLoad> sB = {smem + BLOCK_M * BLOCK_K};
    using ATile = MFMAF32_16x16F32_ATile<InnerK>;
    using BTile = MFMAF32_16x16F32_BTile<InnerK>;
    using CTile = MFMAF32_16x16F32_CTile;
    using Mma = TileMMA<ATile, BTile, CTile>;
    using GemmInstance = GemmBlockT_V1<BLOCK_M, BLOCK_K, BLOCK_N, ATile, BTile, CTile, Mma, Warps>;

    using LdgA = LdgBlockFrag<BLOCK_M, BLOCK_K, Warps, VecLoad>;
    using LdgB = LdgBlockFrag<BLOCK_K, BLOCK_N, Warps, VecLoad>;
    LdgA ldg_a;
    LdgB ldg_b;
    GemmInstance block_gemm;
    block_gemm.fill_c(0.0f);

    for (int k = 0; k < cdiv(K, BLOCK_K); ++k) {
        ldg_a.copy_g2r(&gA);
        ldg_b.copy_g2r(&gB);
        ldg_a.copy_r2s(&sA);
        ldg_b.copy_r2s(&sB);
        __syncthreads();
        block_gemm.mma(&sA, &sB);
        __syncthreads();
    }

    GMemTile<float> gC = {C, M, N, offset_m, offset_n};
    block_gemm.copy_r2g(&gC);
}