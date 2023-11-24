#include "hip_utils.hpp"
#include "mfma_tools.cpp"
#include <hip/amd_detail/amd_hip_runtime.h>
#include <stdio.h>
#include <tuple>


/// block level
template <int BLOCK_A, int BLOCK_B, int Warps, int VecLoad>
struct LdgBlockFrag {
    static_assert(is_power_of_2(BLOCK_B) && BLOCK_B % VecLoad == 0, "");
    static_assert(is_power_of_2(BLOCK_A), "");
    static_assert(BLOCK_B <= Warps * warp_size, "not enough threads per row");
    static constexpr int threads_per_row = BLOCK_B / VecLoad;
    static constexpr int stride_col = Warps * warp_size / threads_per_row;
    static constexpr int rep_m = BLOCK_A / stride_col;
    static constexpr bool oversub = stride_col > BLOCK_A;
    
    float ldg_regs[Max<rep_m, 1>::value][VecLoad];

    template <typename GmemAcc>
    __device__ void copy_g2r(GmemAcc& gA) {
        int row = threadIdx.x % threads_per_row;
        int col = threadIdx.x / threads_per_row;
        if constexpr(oversub) {
            if (col < BLOCK_A) {
                auto ptr = gA.index(col, row * VecLoad);
                vec_load_nullable<VecLoad>(ptr, &ldg_regs[0][0]);
            }
        } else {
            for (int i = 0; i < rep_m; ++i) {
                auto ptr = gA.index(i * stride_col + col, row * VecLoad);
                // if is null, then fill with zeros, null represents out of bounds
                vec_load_nullable<VecLoad>(ptr, &ldg_regs[i][0]);
            }
        }
    }

    template <typename SmemAcc>
    __device__ void copy_r2s(SmemAcc& sA) {
        int row = threadIdx.x % threads_per_row;
        int col = threadIdx.x / threads_per_row;
        if constexpr(oversub) {
            if (col < BLOCK_A) {
                for (int j = 0; j < VecLoad; ++j)
                    *sA.index(col, row * VecLoad + j) = ldg_regs[0][j];
            }
        } else {
            for (int i = 0; i < rep_m; ++i)
                // assume that the compiler can detect consecutive memory accesses and vectorize them
                for (int j = 0; j < VecLoad; ++j)
                    *sA.index(i * stride_col + col, row * VecLoad + j) = ldg_regs[i][j];            
        }
    }

    __device__ void print_frag() {
        __syncthreads();
        if (threadIdx.x == 0) {
            printf("config: reps: %d, cols %d, rows %d, oversub %d, vecload %d\n", rep_m, stride_col, threads_per_row, oversub, VecLoad);
            printf("block_a %d, block_b %d\n", BLOCK_A, BLOCK_B);
        }
        repeat<rep_m, stride_col, threads_per_row>([&](int im, int sc, int tr) {
            if (threadIdx.x == sc * threads_per_row + tr)
                for (int i = 0; i < VecLoad; ++i)
                    printf("%f ", ldg_regs[im][i]);
            __syncthreads();
            if (((sc == stride_col - 1 && stride_col > 1) || (im == rep_m - 1 && rep_m > 1) || (tr == threads_per_row - 1)) && threadIdx.x == 0) {
                printf("\n");
            }
            __syncthreads();
        });
        __syncthreads();
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
    static_assert(is_power_of_2(BLOCK_K) && BLOCK_K >= mma_k);

    float regs[rep_k];
    template <typename SmemAcc>
    __device__ void copy_s2r(SmemAcc& sA) {
        int lane = threadIdx.x % warp_size;
        if constexpr(rep_k % 4 == 0) {
            for (int i = 0; i < rep_k / 4; ++i) {
                for (int j = 0; j < 4; ++j) {
                    regs[i * 4 + j] = *sA.index((lane % mma_m), (lane / mma_m + i * mma_k) * 4 + j);
                }
            }
        } else {
            for (int i = 0; i < rep_k; ++i) {
                regs[i] = *sA.index((lane % mma_m), (lane / mma_m) + i * mma_k);
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
    static_assert(is_power_of_2(BLOCK_K) && BLOCK_K >= mma_k);

    float regs[rep_k];
    template <typename SmemAcc>
    __device__ void copy_s2r(SmemAcc& sB) {
        int lane = threadIdx.x % warp_size;
        if constexpr(rep_k % 4 == 0) {
            for (int i = 0; i < rep_k / 4; ++i) {
                for (int j = 0; j < 4; ++j) {
                    regs[i * 4 + j] = *sB.index((lane / mma_n + i * mma_k) * 4 + j, lane % mma_n);
                }
            }
        } else {
            for (int i = 0; i < rep_k; ++i) {
                regs[i] = *sB.index((lane / mma_n) + i * mma_k, lane % mma_n);
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
    __device__ void copy_r2g(GMemC &gC) {
        int lane = threadIdx.x % warp_size;
        for (int i = 0; i < 4; ++i) {
            auto ptr = gC.index((lane / 16) * 4 + i, lane % 16);
            if (ptr != nullptr)
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
            OffsetAccessor<GMemC> gC_acc = {gC, warp_m * rep_m * tile_m + i * tile_m, warp_n * rep_n * tile_n + j * tile_n};
            Ctiles[i][j].copy_r2g(gC_acc);
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
        int warp_n = warp / super::warps_m;

        for (int k = 0; k < super::rep_k; k++) {
            // load m
            for (int im = 0; im < super::rep_m; im++) {
                OffsetAccessor<SMemA> sA_acc = {
                    sA, 
                    warp_m * super::tile_m * super::rep_m + im * super::tile_m,
                    k * super::tile_k    
                };

                Atiles[im].copy_s2r(sA_acc);
            }

            // load n
            for (int in = 0; in < super::rep_n; in++) {
                OffsetAccessor<SMemB> sB_acc = {
                    sB,
                    k * super::tile_k,
                    warp_n * super::tile_n * super::rep_n + in * super::tile_n
                };

                Btiles[in].copy_s2r(sB_acc);
            }

            // mma
            repeat<super::rep_m, super::rep_n>([&](int i, int j) {
                MMA::mma(Atiles[i], Btiles[j], this->Ctiles[i][j]);
            });
        }
    }
};

template <int BLOCK_M, int BLOCK_K, int BLOCK_N, int VecLoad, int InnerK, int Warps>
struct Mfma_gemmv3 : BasicGemmInstance<BLOCK_M, BLOCK_K, BLOCK_N, Warps> {
    using BasicGemmInstance<BLOCK_M, BLOCK_K, BLOCK_N, Warps>::BasicGemmInstance;

    using super = BasicGemmInstance<BLOCK_M, BLOCK_K, BLOCK_N, Warps>;

    __device__ void run() {
        const int offset_m = blockIdx.x * BLOCK_M;
        const int offset_n = blockIdx.y * BLOCK_N;


        // auto gA = OffsetAccessor<RowAccessor<const float>>(RowAccessor<const float>(this->a, this->m, this->k), offset_m, 0);
        RowAccessor<const float> gA_ = {this->a, this->m, this->k};
        OffsetAccessor<RowAccessor<const float>> gA = {gA_, offset_m, 0};
        RowAccessor<const float> gB_ = {this->b, this->k, this->n};
        OffsetAccessor<RowAccessor<const float>> gB = {gB_, 0, offset_n};
        // GMemTile<const float> gB = {this->b, this->k, this->n, 0, offset_n};

        static_assert(BLOCK_K % VecLoad == 0, "");
        static_assert(BLOCK_K % InnerK == 0, "");
        using SharedMemLayoutA = ComposeLayout<SwizzleLayout<BLOCK_M, BLOCK_K / VecLoad>, RowLayout<1, VecLoad>>;
        // using SharedMemLayoutA = RowLayout<BLOCK_M, BLOCK_K>;
        auto sA = LayoutAccessor<SharedMemLayoutA, float>(this->smem);

        using SharedMemLayoutB = TransposeLayout<ComposeLayout<SwizzleLayout<BLOCK_N, BLOCK_K / VecLoad>, RowLayout<1, VecLoad>>>;
        // using SharedMemLayoutB = RowLayout<BLOCK_K, BLOCK_N>;
        auto sB = LayoutAccessor<SharedMemLayoutB, float>(this->smem + BLOCK_M * BLOCK_K);

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

        for (int k = 0; k < cdiv(this->k, BLOCK_K); ++k) {
            ldg_a.copy_g2r(gA);
            ldg_b.copy_g2r(gB);
            ldg_a.copy_r2s(sA);
            ldg_b.copy_r2s(sB);
            // __syncthreads();
            // ldg_a.print_frag();
            // __syncthreads();
            // print_smem(sA);
            // print_smem(sB);
            __syncthreads();
            block_gemm.mma(sA, sB);
            __syncthreads();
            gA.inc_offset(0, BLOCK_K);
            gB.inc_offset(BLOCK_K, 0);

        }
        RowAccessor<float> gC_ = {this->c, this->m, this->n};
        OffsetAccessor<RowAccessor<float>> gC = {gC_, offset_m, offset_n};
        // GMemTile<float> gC = {this->c, this->m, this->n, offset_m, offset_n};
        block_gemm.copy_r2g(gC);
    }
};

#include "launch_defs.hpp"

#ifndef _VecLoad
#define _VecLoad 1
#endif

#ifndef _InnerK
#define _InnerK 4
#endif


EXPORT bool LAUNCH_NAME(
    const float* A, const float* B, float* C,
    int M, int K, int N
) {
    using GemmInstance = Mfma_gemmv3<_BLOCK_M, _BLOCK_K, _BLOCK_N, _VecLoad, _InnerK, _Warps>;
    return run_kernel<GemmInstance>(A, B, C, M, K, N);

}