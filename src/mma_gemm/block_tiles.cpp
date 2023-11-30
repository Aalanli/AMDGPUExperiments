#pragma once

#include "hip_utils.hpp"
#include "layouts.cpp"

/// block level
template <typename T, int BLOCK_A, int BLOCK_B, int Warps, int VecLoad>
struct LdgBlockFrag {
    static_assert(is_power_of_2(BLOCK_B) && BLOCK_B % VecLoad == 0, "");
    static_assert(is_power_of_2(BLOCK_A), "");
    static_assert(BLOCK_B <= Warps * warp_size, "not enough threads per row");
    static constexpr int threads_per_row = BLOCK_B / VecLoad;
    static constexpr int stride_col = Warps * warp_size / threads_per_row;
    static constexpr int rep_m = BLOCK_A / stride_col;
    static constexpr bool oversub = stride_col > BLOCK_A;
    
    T ldg_regs[Max<rep_m, 1>::value][VecLoad];

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
                    printf("%f ", (float) ldg_regs[im][i]);
            __syncthreads();
            if (((sc == stride_col - 1 && stride_col > 1) || (im == rep_m - 1 && rep_m > 1) || (tr == threads_per_row - 1)) && threadIdx.x == 0) {
                printf("\n");
            }
            __syncthreads();
        });
        __syncthreads();
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
struct BlockGemmBase {
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
struct BlockGemmV1 : BlockGemmBase<BLOCK_M, BLOCK_K, BLOCK_N, ATile, BTile, CTile, Warps> {
    using super = BlockGemmBase<BLOCK_M, BLOCK_K, BLOCK_N, ATile, BTile, CTile, Warps>;

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

template <int BLOCK_M, 
          int BLOCK_K, 
          int BLOCK_N, 
          typename ATile, 
          typename BTile, 
          typename CTile, 
          typename MMA,
          int Warps>
struct BlockGemmV1_Init : BlockGemmBase<BLOCK_M, BLOCK_K, BLOCK_N, ATile, BTile, CTile, Warps> {
    using super = BlockGemmBase<BLOCK_M, BLOCK_K, BLOCK_N, ATile, BTile, CTile, Warps>;

    ATile Atiles[super::rep_m];
    BTile Btiles[super::rep_n];

    template <typename SMemA, typename SMemB>
    __device__ void mma(SMemA &sA, SMemB &sB) {
        int warp = __builtin_amdgcn_readfirstlane(threadIdx.x / warp_size);
        int warp_m = warp % super::warps_m;
        int warp_n = warp / super::warps_m;
        
        int offset_k = 0;
        #pragma unroll
        for (int k = 0; k < super::rep_k; k++) {
            // load m
            #pragma unroll
            for (int im = 0; im < super::rep_m; im++) {
                OffsetAccessor<SMemA> sA_acc = {
                    sA, 
                    warp_m * super::tile_m * super::rep_m + im * super::tile_m,
                    offset_k    
                };

                Atiles[im].copy_s2r(sA_acc);
            }

            // load n
            #pragma unroll
            for (int in = 0; in < super::rep_n; in++) {
                OffsetAccessor<SMemB> sB_acc = {
                    sB,
                    offset_k,
                    warp_n * super::tile_n * super::rep_n + in * super::tile_n
                };

                Btiles[in].copy_s2r(sB_acc);
            }
            offset_k += super::tile_k;

            // mma
            repeat<super::rep_m, super::rep_n>([&](int i, int j) {
                MMA::mma(Atiles[i], Btiles[j], this->Ctiles[i][j]);
            });
        }
    }
};

/// pipeline loading from smem
template <int BLOCK_M, 
          int BLOCK_K, 
          int BLOCK_N, 
          typename ATile, 
          typename BTile, 
          typename CTile, 
          typename MMA,
          int Warps>
struct BlockGemmV2 : BlockGemmBase<BLOCK_M, BLOCK_K, BLOCK_N, ATile, BTile, CTile, Warps> {
    using super = BlockGemmBase<BLOCK_M, BLOCK_K, BLOCK_N, ATile, BTile, CTile, Warps>;

    ATile Atiles[2][super::rep_m];
    BTile Btiles[2][super::rep_n];

    template <typename SMemA>
    __device__ void load_atile(SMemA &sA, int idx, int k_offset) {
        // lol who cares about DRY?
        int warp = threadIdx.x / warp_size;
        int warp_m = warp % super::warps_m;
        int warp_n = warp / super::warps_m;

        for (int im = 0; im < super::rep_m; im++) {
            OffsetAccessor<SMemA> sA_acc = {
                sA, 
                warp_m * super::tile_m * super::rep_m + im * super::tile_m,
                k_offset    
            };

            Atiles[idx][im].copy_s2r(sA_acc);
        }
    }

    template <typename SMemB>
    __device__ void load_btile(SMemB &sB, int idx, int k_offset) {
        int warp = threadIdx.x / warp_size;
        int warp_m = warp % super::warps_m;
        int warp_n = warp / super::warps_m;

        for (int in = 0; in < super::rep_n; in++) {
            OffsetAccessor<SMemB> sB_acc = {
                sB,
                k_offset,
                warp_n * super::tile_n * super::rep_n + in * super::tile_n
            };

            Btiles[idx][in].copy_s2r(sB_acc);
        }
    }

    template <typename SMemA, typename SMemB>
    __device__ void mma(SMemA &sA, SMemB &sB) {
        int warp = threadIdx.x / warp_size;
        int warp_m = warp % super::warps_m;
        int warp_n = warp / super::warps_m;

        load_atile(sA, 0, 0);
        load_btile(sB, 0, 0);

        #pragma unroll
        for (int k = 0; k < super::rep_k; k++) {
            if (k < super::rep_k - 1) {
                load_atile(sA, (k + 1) % 2, (k + 1) * super::tile_k);
                load_btile(sB, (k + 1) % 2, (k + 1) * super::tile_k);
            }
            repeat<super::rep_m, super::rep_n>([&](int i, int j) {
                MMA::mma(Atiles[k % 2][i], Btiles[k % 2][j], this->Ctiles[i][j]);
            });
        }
    }
};