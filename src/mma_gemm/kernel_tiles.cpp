#pragma once

#include "hip_utils.hpp"
#include "warp_tiles.cpp"
#include "block_tiles.cpp"
#include <__clang_hip_runtime_wrapper.h>
#include <hip/amd_detail/amd_hip_runtime.h>


template <typename T, int BLOCK_M, int BLOCK_K, int BLOCK_N, int Warps>
struct BasicGemmInstance {
    T* smem;
    const T* __restrict__ a;
    const T* __restrict__ b;
    T* __restrict__ c;
    const int m, n, k;
    static constexpr int used_smem_bytes() {
        return (BLOCK_M + BLOCK_N) * BLOCK_K * sizeof(T);
    }

    static constexpr int nthreads() {
        return Warps * warp_size;
    }

    DevHost BasicGemmInstance(
        T* smem, const T* a, const T* b, T* c, int m, int n, int k
    ) : smem(smem), a(a), b(b), c(c), m(m), n(n), k(k) {}

    static DevHost dim3 blocks(int m, int k, int n) {
        return dim3((m + BLOCK_M - 1) / BLOCK_M, (n + BLOCK_N - 1) / BLOCK_N);
    }

    static DevHost dim3 threads() {
        return dim3(nthreads());
    }
    
    __device__ inline void run() {
        assert(false && "NotImplemented");
    }
};


template <typename T, int BLOCK_M, int BLOCK_K, int BLOCK_N, int VecLoad, int InnerK, int Warps>
struct Mfma_gemmv3_ref : BasicGemmInstance<T, BLOCK_M, BLOCK_K, BLOCK_N, Warps> {
    using BasicGemmInstance<T, BLOCK_M, BLOCK_K, BLOCK_N, Warps>::BasicGemmInstance;

    using super = BasicGemmInstance<T, BLOCK_M, BLOCK_K, BLOCK_N, Warps>;

    __device__ inline void run() {
        const int offset_m = blockIdx.x * BLOCK_M;
        const int offset_n = blockIdx.y * BLOCK_N;


        // auto gA = OffsetAccessor<RowAccessor<const T>>(RowAccessor<const T>(this->a, this->m, this->k), offset_m, 0);
        RowAccessor<const T> gA_ = {this->a, this->m, this->k};
        OffsetAccessor<RowAccessor<const T>> gA = {gA_, offset_m, 0};
        RowAccessor<const T> gB_ = {this->b, this->k, this->n};
        OffsetAccessor<RowAccessor<const T>> gB = {gB_, 0, offset_n};
        // GMemTile<const T> gB = {this->b, this->k, this->n, 0, offset_n};

        static_assert(BLOCK_K % VecLoad == 0, "");
        static_assert(BLOCK_K % InnerK == 0, "");
        using SharedMemLayoutA = ComposeLayout<SwizzleLayout<BLOCK_M, BLOCK_K / VecLoad>, RowLayout<1, VecLoad>>;
        // using SharedMemLayoutA = RowLayout<BLOCK_M, BLOCK_K>;
        auto sA = LayoutAccessor<SharedMemLayoutA, T>(this->smem);

        using SharedMemLayoutB = TransposeLayout<ComposeLayout<SwizzleLayout<BLOCK_N, BLOCK_K / VecLoad>, RowLayout<1, VecLoad>>>;
        // using SharedMemLayoutB = RowLayout<BLOCK_K, BLOCK_N>;
        auto sB = LayoutAccessor<SharedMemLayoutB, T>(this->smem + BLOCK_M * BLOCK_K);

        using ATile = MFMAF32_16x16F32_ATile<InnerK>;
        using BTile = MFMAF32_16x16F32_BTile<InnerK>;
        using CTile = MFMAF32_16x16F32_CTile;
        using Mma = TileMMA<ATile, BTile, CTile>;
        using GemmInstance = BlockGemmV1<BLOCK_M, BLOCK_K, BLOCK_N, ATile, BTile, CTile, Warps>;

        using LdgA = LdgBlockFrag<T, BLOCK_M, BLOCK_K, Warps, VecLoad>;
        using LdgB = LdgBlockFrag<T, BLOCK_K, BLOCK_N, Warps, VecLoad>;
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
        RowAccessor<T> gC_ = {this->c, this->m, this->n};
        OffsetAccessor<RowAccessor<T>> gC = {gC_, offset_m, offset_n};
        // GMemTile<T> gC = {this->c, this->m, this->n, offset_m, offset_n};
        block_gemm.copy_r2g(gC);
    }
};


template <typename T,
          int BLOCK_M, 
          int BLOCK_K, 
          int BLOCK_N, 
          int VecLoad, 
          int InnerK, 
          int Warps,
          typename SharedMemLayoutA,
          typename SharedMemLayoutB,
          typename GemmInstance>
struct Mfma_gemmv3 : BasicGemmInstance<T, BLOCK_M, BLOCK_K, BLOCK_N, Warps> {
    using BasicGemmInstance<T, BLOCK_M, BLOCK_K, BLOCK_N, Warps>::BasicGemmInstance;

    using super = BasicGemmInstance<T, BLOCK_M, BLOCK_K, BLOCK_N, Warps>;

    static constexpr int used_smem_bytes() {
        return sizeof(T) * (SharedMemLayoutA::s1 * SharedMemLayoutA::s2 + SharedMemLayoutB::s1 * SharedMemLayoutB::s2);
    }

    __device__ inline void run() {
        const int offset_m = blockIdx.x * BLOCK_M;
        const int offset_n = blockIdx.y * BLOCK_N;

        RowAccessor<const T> gA_ = {this->a, this->m, this->k};
        OffsetAccessor<RowAccessor<const T>> gA = {gA_, offset_m, 0};
        RowAccessor<const T> gB_ = {this->b, this->k, this->n};
        OffsetAccessor<RowAccessor<const T>> gB = {gB_, 0, offset_n};

        static_assert(BLOCK_K % VecLoad == 0, "");
        static_assert(BLOCK_K % InnerK == 0, "");

        // static_assert(SharedMemLayoutA::s1 == BLOCK_M && SharedMemLayoutA::s2 == BLOCK_K, "");
        auto sA = LayoutAccessor<SharedMemLayoutA, T>(this->smem);

        // static_assert(SharedMemLayoutB::s1 == BLOCK_K && SharedMemLayoutB::s2 == BLOCK_N, "");
        auto sB = LayoutAccessor<SharedMemLayoutB, T>(this->smem + SharedMemLayoutA::s1 * SharedMemLayoutA::s2);

        static_assert(
            GemmInstance::block_m == BLOCK_M && 
            GemmInstance::block_n == BLOCK_N && 
            GemmInstance::block_k == BLOCK_K, ""
        );

        using LdgA = LdgBlockFrag<T, BLOCK_M, BLOCK_K, Warps, VecLoad>;
        using LdgB = LdgBlockFrag<T, BLOCK_K, BLOCK_N, Warps, VecLoad>;
        LdgA ldg_a;
        LdgB ldg_b;
        GemmInstance block_gemm;
        block_gemm.fill_c(0.0f);

        for (int k = 0; k < cdiv(this->k, BLOCK_K); ++k) {
            ldg_a.copy_g2r(gA);
            ldg_b.copy_g2r(gB);
            ldg_a.copy_r2s(sA);
            ldg_b.copy_r2s(sB);
            __syncthreads();
            block_gemm.mma(sA, sB);
            __syncthreads();
            gA.inc_offset(0, BLOCK_K);
            gB.inc_offset(BLOCK_K, 0);

        }
        RowAccessor<T> gC_ = {this->c, this->m, this->n};
        OffsetAccessor<RowAccessor<T>> gC = {gC_, offset_m, offset_n};
        block_gemm.copy_r2g(gC);
    }
};


template <typename T,
          int BLOCK_M, 
          int BLOCK_K, 
          int BLOCK_N, 
          int VecLoad, 
          int InnerK, 
          int Warps,
          typename SharedMemLayoutA,
          typename SharedMemLayoutB,
          typename GemmInstance>
struct Mfma_gemmv3_Ldgv2 : BasicGemmInstance<T, BLOCK_M, BLOCK_K, BLOCK_N, Warps> {
    using BasicGemmInstance<T, BLOCK_M, BLOCK_K, BLOCK_N, Warps>::BasicGemmInstance;

    using super = BasicGemmInstance<T, BLOCK_M, BLOCK_K, BLOCK_N, Warps>;

    static constexpr int used_smem_bytes() {
        return sizeof(T) * (SharedMemLayoutA::s1 * SharedMemLayoutA::s2 + SharedMemLayoutB::s1 * SharedMemLayoutB::s2);
    }

    __device__ inline void run() {
        const int offset_m = blockIdx.x * BLOCK_M;
        const int offset_n = blockIdx.y * BLOCK_N;

        static_assert(BLOCK_K % VecLoad == 0, "");
        static_assert(BLOCK_K % InnerK == 0, "");

        // static_assert(SharedMemLayoutA::s1 == BLOCK_M && SharedMemLayoutA::s2 == BLOCK_K, "");
        auto sA = LayoutAccessor<SharedMemLayoutA, T>(this->smem);

        // static_assert(SharedMemLayoutB::s1 == BLOCK_K && SharedMemLayoutB::s2 == BLOCK_N, "");
        auto sB = LayoutAccessor<SharedMemLayoutB, T>(this->smem + SharedMemLayoutA::s1 * SharedMemLayoutA::s2);

        static_assert(
            GemmInstance::block_m == BLOCK_M && 
            GemmInstance::block_n == BLOCK_N && 
            GemmInstance::block_k == BLOCK_K, ""
        );

        LdgBlockFragv2<T, BLOCK_M, BLOCK_K, Warps, VecLoad> ldg_a(this->a, offset_m, 0, this->m, this->k);
        LdgBlockFragv2<T, BLOCK_K, BLOCK_N, Warps, VecLoad> ldg_b(this->b, 0, offset_n, this->k, this->n);
        
        GemmInstance block_gemm;
        block_gemm.fill_c(0.0f);

        for (int k = 0; k < cdiv(this->k, BLOCK_K); ++k) {
            ldg_a.copy_g2r();
            ldg_b.copy_g2r();
            ldg_a.copy_r2s(sA);
            ldg_b.copy_r2s(sB);
            __syncthreads();
            // print_acc(sA);

            // __syncthreads();
            block_gemm.mma(sA, sB);
            __syncthreads();
            ldg_a.inc_offset(0, BLOCK_K);
            ldg_b.inc_offset(BLOCK_K, 0);

        }
        RowAccessor<T> gC_ = {this->c, this->m, this->n};
        OffsetAccessor<RowAccessor<T>> gC = {gC_, offset_m, offset_n};
        block_gemm.copy_r2g(gC);
    }
};

template <typename T,
          int BLOCK_M, 
          int BLOCK_K, 
          int BLOCK_N, 
          int VecLoad, 
          int InnerK, 
          int Warps,
          typename SharedMemLayoutA,
          typename SharedMemLayoutB,
          typename GemmInstance>
struct Mfma_gemmv3_Pipeline1 : BasicGemmInstance<T, BLOCK_M, BLOCK_K, BLOCK_N, Warps> {
    using BasicGemmInstance<T, BLOCK_M, BLOCK_K, BLOCK_N, Warps>::BasicGemmInstance;

    using super = BasicGemmInstance<T, BLOCK_M, BLOCK_K, BLOCK_N, Warps>;
    static constexpr int used_smem_bytes() {
        return sizeof(T) * (SharedMemLayoutA::s1 * SharedMemLayoutA::s2 + SharedMemLayoutB::s1 * SharedMemLayoutB::s2);
    }

    __device__ inline void run() {
        const int offset_m = blockIdx.x * BLOCK_M;
        const int offset_n = blockIdx.y * BLOCK_N;

        RowAccessor<const T> gA_ = {this->a, this->m, this->k};
        OffsetAccessor<RowAccessor<const T>> gA = {gA_, offset_m, 0};
        RowAccessor<const T> gB_ = {this->b, this->k, this->n};
        OffsetAccessor<RowAccessor<const T>> gB = {gB_, 0, offset_n};

        static_assert(BLOCK_K % VecLoad == 0, "");
        static_assert(BLOCK_K % InnerK == 0, "");

        // static_assert(SharedMemLayoutA::s1 == BLOCK_M && SharedMemLayoutA::s2 == BLOCK_K, "");
        auto sA = LayoutAccessor<SharedMemLayoutA, T>(this->smem);

        // static_assert(SharedMemLayoutB::s1 == BLOCK_K && SharedMemLayoutB::s2 == BLOCK_N, "");
        auto sB = LayoutAccessor<SharedMemLayoutB, T>(this->smem + SharedMemLayoutA::s1 * SharedMemLayoutA::s2);

        static_assert(
            GemmInstance::block_m == BLOCK_M && 
            GemmInstance::block_n == BLOCK_N && 
            GemmInstance::block_k == BLOCK_K, ""
        );

        using LdgA = LdgBlockFrag<T, BLOCK_M, BLOCK_K, Warps, VecLoad>;
        using LdgB = LdgBlockFrag<T, BLOCK_K, BLOCK_N, Warps, VecLoad>;
        LdgA ldg_a;
        LdgB ldg_b;
        GemmInstance block_gemm;
        block_gemm.fill_c(0.0f);

        ldg_a.copy_g2r(gA);
        ldg_b.copy_g2r(gB);
        ldg_a.copy_r2s(sA);
        ldg_b.copy_r2s(sB);
        gA.inc_offset(0, BLOCK_K);
        gB.inc_offset(BLOCK_K, 0);

        for (int k = 0; k < cdiv(this->k, BLOCK_K) - 1; ++k) {
            ldg_a.copy_g2r(gA);
            // compiler bug I think: when vectorized pack=8, block_n = 32, block_m = 32, block_k = 16, warps = 2, inner_k = 16
            //  uncommenting below removes error
            
            // __syncthreads();
            // if (blockIdx.x == 0 && blockIdx.y == 0) {
            //     if (threadIdx.x == 0) {
            //         // repeat<BLOCK_M, BLOCK_K>([&](int i, int j) {
            //         //     printf("%f ", (float) *sA.index(i, j));
            //         //     if (j == BLOCK_K - 1) {
            //         //         printf("\n");
            //         //     }
            //         // });
            //         printf("\n");
            //     }
            // }
            __syncthreads();
            ldg_b.copy_g2r(gB);
            block_gemm.mma(sA, sB);
            __syncthreads();

            gA.inc_offset(0, BLOCK_K);
            gB.inc_offset(BLOCK_K, 0);
            ldg_a.copy_r2s(sA);
            ldg_b.copy_r2s(sB);

        }
        __syncthreads();
        block_gemm.mma(sA, sB);

        RowAccessor<T> gC_ = {this->c, this->m, this->n};
        OffsetAccessor<RowAccessor<T>> gC = {gC_, offset_m, offset_n};
        block_gemm.copy_r2g(gC);
    }
};


template <typename T,
          int BLOCK_M, 
          int BLOCK_K, 
          int BLOCK_N, 
          int VecLoad, 
          int InnerK, 
          int Warps,
          typename SharedMemLayoutA,
          typename SharedMemLayoutB,
          typename GemmInstance>
struct Mfma_gemmv3_Pipeline1_E : BasicGemmInstance<T, BLOCK_M, BLOCK_K, BLOCK_N, Warps> {
    using BasicGemmInstance<T, BLOCK_M, BLOCK_K, BLOCK_N, Warps>::BasicGemmInstance;

    using super = BasicGemmInstance<T, BLOCK_M, BLOCK_K, BLOCK_N, Warps>;
    static constexpr int used_smem_bytes() {
        return sizeof(T) * (SharedMemLayoutA::s1 * SharedMemLayoutA::s2 + SharedMemLayoutB::s1 * SharedMemLayoutB::s2);
    }
    
    __device__ inline void run() {
        const int offset_m = blockIdx.x * BLOCK_M;
        const int offset_n = blockIdx.y * BLOCK_N;

        RowAccessor<const T> gA_ = {this->a, this->m, this->k};
        OffsetAccessor<RowAccessor<const T>> gA = {gA_, offset_m, 0};
        RowAccessor<const T> gB_ = {this->b, this->k, this->n};
        OffsetAccessor<RowAccessor<const T>> gB = {gB_, 0, offset_n};

        static_assert(BLOCK_K % VecLoad == 0, "");
        static_assert(BLOCK_K % InnerK == 0, "");

        // static_assert(SharedMemLayoutA::s1 == BLOCK_M && SharedMemLayoutA::s2 == BLOCK_K, "");
        auto sA = LayoutAccessor<SharedMemLayoutA, T>(this->smem);

        // static_assert(SharedMemLayoutB::s1 == BLOCK_K && SharedMemLayoutB::s2 == BLOCK_N, "");
        auto sB = LayoutAccessor<SharedMemLayoutB, T>(this->smem + SharedMemLayoutA::s1 * SharedMemLayoutA::s2);

        static_assert(
            GemmInstance::block_m == BLOCK_M && 
            GemmInstance::block_n == BLOCK_N && 
            GemmInstance::block_k == BLOCK_K, ""
        );

        using LdgA = LdgBlockFrag<T, BLOCK_M, BLOCK_K, Warps, VecLoad>;
        using LdgB = LdgBlockFrag<T, BLOCK_K, BLOCK_N, Warps, VecLoad>;
        LdgA ldg_a;
        LdgB ldg_b;
        GemmInstance block_gemm;
        ldg_a.copy_g2r(gA);
        ldg_b.copy_g2r(gB);
        gA.inc_offset(0, BLOCK_K);
        gB.inc_offset(BLOCK_K, 0);
        block_gemm.fill_c(0.0f);

        ldg_a.copy_r2s(sA);
        ldg_b.copy_r2s(sB);

        const int ktiles = cdiv(this->k, BLOCK_K) - 1;
        for (int k = 0; k < ktiles; ++k) {
            ldg_a.copy_g2r(gA);
            block_sync_lds();

            ldg_b.copy_g2r(gB);
            block_gemm.mma(sA, sB);

            block_sync_lds();
            gA.inc_offset(0, BLOCK_K);
            gB.inc_offset(BLOCK_K, 0);

            ldg_a.copy_r2s(sA);
            ldg_b.copy_r2s(sB);        
        }
        block_sync_lds();
        block_gemm.mma(sA, sB);

        RowAccessor<T> gC_ = {this->c, this->m, this->n};
        OffsetAccessor<RowAccessor<T>> gC = {gC_, offset_m, offset_n};
        block_gemm.copy_r2g(gC);
    }
};



template <typename T,
          int BLOCK_M, 
          int BLOCK_K, 
          int BLOCK_N, 
          int VecLoad, 
          int InnerK, 
          int Warps,
          typename SharedMemLayoutA,
          typename SharedMemLayoutB,
          typename GemmInstance>
struct Mfma_gemmv3_Pipeline1_E_Ldgv2 : BasicGemmInstance<T, BLOCK_M, BLOCK_K, BLOCK_N, Warps> {
    using BasicGemmInstance<T, BLOCK_M, BLOCK_K, BLOCK_N, Warps>::BasicGemmInstance;

    using super = BasicGemmInstance<T, BLOCK_M, BLOCK_K, BLOCK_N, Warps>;
    static constexpr int used_smem_bytes() {
        return sizeof(T) * (SharedMemLayoutA::s1 * SharedMemLayoutA::s2 + SharedMemLayoutB::s1 * SharedMemLayoutB::s2);
    }
    
    __device__ inline void run() {
        const int offset_m = blockIdx.x * BLOCK_M;
        const int offset_n = blockIdx.y * BLOCK_N;

        static_assert(BLOCK_K % VecLoad == 0, "");
        static_assert(BLOCK_K % InnerK == 0, "");

        // static_assert(SharedMemLayoutA::s1 == BLOCK_M && SharedMemLayoutA::s2 == BLOCK_K, "");
        auto sA = LayoutAccessor<SharedMemLayoutA, T>(this->smem);

        // static_assert(SharedMemLayoutB::s1 == BLOCK_K && SharedMemLayoutB::s2 == BLOCK_N, "");
        auto sB = LayoutAccessor<SharedMemLayoutB, T>(this->smem + SharedMemLayoutA::s1 * SharedMemLayoutA::s2);

        static_assert(
            GemmInstance::block_m == BLOCK_M && 
            GemmInstance::block_n == BLOCK_N && 
            GemmInstance::block_k == BLOCK_K, ""
        );

        LdgBlockFragv2<T, BLOCK_M, BLOCK_K, Warps, VecLoad> ldg_a(this->a, offset_m, 0, this->m, this->k);
        LdgBlockFragv2<T, BLOCK_K, BLOCK_N, Warps, VecLoad> ldg_b(this->b, 0, offset_n, this->k, this->n);
        GemmInstance block_gemm;
        ldg_a.copy_g2r();
        ldg_b.copy_g2r();
        ldg_a.inc_offset(0, BLOCK_K);
        ldg_b.inc_offset(BLOCK_K, 0);
        block_gemm.fill_c(0.0f);

        ldg_a.copy_r2s(sA);
        ldg_b.copy_r2s(sB);

        const int ktiles = cdiv(this->k, BLOCK_K) - 1;
        for (int k = 0; k < ktiles; ++k) {
            ldg_a.copy_g2r();
            block_sync_lds();

            ldg_b.copy_g2r();
            block_gemm.mma(sA, sB);

            block_sync_lds();
            ldg_a.inc_offset(0, BLOCK_K);
            ldg_b.inc_offset(BLOCK_K, 0);

            ldg_a.copy_r2s(sA);
            ldg_b.copy_r2s(sB);        
        }
        block_sync_lds();
        block_gemm.mma(sA, sB);

        RowAccessor<T> gC_ = {this->c, this->m, this->n};
        OffsetAccessor<RowAccessor<T>> gC = {gC_, offset_m, offset_n};
        block_gemm.copy_r2g(gC);
    }
};

template <typename LdgA,
          typename LdgB,
          typename SA,
          typename SB,
          typename GemmInstance,
          int BLOCK_K>
__device__ inline void pipeline_1_E(
    LdgA& ldg_a, LdgB& ldg_b, SA& sA, SB& sB, GemmInstance& block_gemm, int K
) {
    ldg_a.copy_g2r();
    ldg_b.copy_g2r();
    block_gemm.fill_c(0.0f);

    ldg_a.copy_r2s(sA);
    ldg_b.copy_r2s(sB);

    const int ktiles = cdiv(K, BLOCK_K) - 1;
    for (int k = 0; k < ktiles; ++k) {
        ldg_a.copy_g2r();
        block_sync_lds();

        ldg_b.copy_g2r();
        block_gemm.mma(sA, sB);

        block_sync_lds();

        ldg_a.copy_r2s(sA);
        ldg_b.copy_r2s(sB);        
    }
    block_sync_lds();
    block_gemm.mma(sA, sB);
}


template <typename T,
          int BLOCK_M, 
          int BLOCK_K, 
          int BLOCK_N, 
          int VecLoad, 
          int InnerK, 
          int Warps,
          typename SharedMemLayoutA,
          typename SharedMemLayoutB,
          typename GemmInstance>
struct Mfma_gemmv3_Pipeline1_E2 : BasicGemmInstance<T, BLOCK_M, BLOCK_K, BLOCK_N, Warps> {
    using BasicGemmInstance<T, BLOCK_M, BLOCK_K, BLOCK_N, Warps>::BasicGemmInstance;

    using super = BasicGemmInstance<T, BLOCK_M, BLOCK_K, BLOCK_N, Warps>;
    static constexpr int used_smem_bytes() {
        return sizeof(T) * (SharedMemLayoutA::s1 * SharedMemLayoutA::s2 + SharedMemLayoutB::s1 * SharedMemLayoutB::s2);
    }
    
    __device__ inline void run() {
        const int offset_m = __builtin_amdgcn_readfirstlane(blockIdx.x * BLOCK_M);
        const int offset_n = __builtin_amdgcn_readfirstlane(blockIdx.y * BLOCK_N);

        RowAccessor<const T> gA_ = {this->a, this->m, this->k};
        OffsetAccessor<RowAccessor<const T>> gA = {gA_, offset_m, 0};
        RowAccessor<const T> gB_ = {this->b, this->k, this->n};
        OffsetAccessor<RowAccessor<const T>> gB = {gB_, 0, offset_n};

        static_assert(BLOCK_K % VecLoad == 0, "");
        static_assert(BLOCK_K % InnerK == 0, "");

        // static_assert(SharedMemLayoutA::s1 == BLOCK_M && SharedMemLayoutA::s2 == BLOCK_K, "");
        auto sA = LayoutAccessor<SharedMemLayoutA, T>(this->smem);

        // static_assert(SharedMemLayoutB::s1 == BLOCK_K && SharedMemLayoutB::s2 == BLOCK_N, "");
        auto sB = LayoutAccessor<SharedMemLayoutB, T>(this->smem + SharedMemLayoutA::s1 * SharedMemLayoutA::s2);

        static_assert(
            GemmInstance::block_m == BLOCK_M && 
            GemmInstance::block_n == BLOCK_N && 
            GemmInstance::block_k == BLOCK_K, ""
        );

        using LdgA = LdgBlockFrag<T, BLOCK_M, BLOCK_K, Warps, VecLoad>;
        using LdgB = LdgBlockFrag<T, BLOCK_K, BLOCK_N, Warps, VecLoad>;
        LdgA ldg_a;
        LdgB ldg_b;
        GemmInstance block_gemm;
        ldg_a.copy_g2r(gA);
        ldg_b.copy_g2r(gB);
        gA.inc_offset(0, BLOCK_K);
        gB.inc_offset(BLOCK_K, 0);
        block_gemm.fill_c(0.0f);

        ldg_a.copy_r2s(sA);
        ldg_b.copy_r2s(sB);

        const int ktiles = cdiv(this->k, BLOCK_K) - 1;
        for (int k = 0; k < ktiles; ++k) {
            ldg_a.copy_g2r(gA);
            block_sync_lds();

            ldg_b.copy_g2r(gB);
            block_gemm.mma(sA, sB);

            block_sync_lds();
            gA.inc_offset(0, BLOCK_K);
            gB.inc_offset(BLOCK_K, 0);

            ldg_a.copy_r2s(sA);
            ldg_b.copy_r2s(sB);        
        }
        block_sync_lds();
        block_gemm.mma(sA, sB);

        RowAccessor<T> gC_ = {this->c, this->m, this->n};
        OffsetAccessor<RowAccessor<T>> gC = {gC_, offset_m, offset_n};
        block_gemm.copy_r2g(gC);
    }
};




template <typename T,
          int BLOCK_M, 
          int BLOCK_K, 
          int BLOCK_N, 
          int VecLoad, 
          int InnerK, 
          int Warps,
          typename SharedMemLayoutA,
          typename SharedMemLayoutB,
          typename GemmInstance>
struct Mfma_gemmv3_Pipeline2 : BasicGemmInstance<T, BLOCK_M, BLOCK_K, BLOCK_N, Warps> {
    using BasicGemmInstance<T, BLOCK_M, BLOCK_K, BLOCK_N, Warps>::BasicGemmInstance;

    using super = BasicGemmInstance<T, BLOCK_M, BLOCK_K, BLOCK_N, Warps>;

    static constexpr int used_smem_bytes() {
        return 2 * (BLOCK_M + BLOCK_N) * BLOCK_K * sizeof(T);
    }

    __device__ inline void run() {
        const int offset_m = blockIdx.x * BLOCK_M;
        const int offset_n = blockIdx.y * BLOCK_N;

        RowAccessor<const T> gA_ = {this->a, this->m, this->k};
        OffsetAccessor<RowAccessor<const T>> gA = {gA_, offset_m, 0};
        RowAccessor<const T> gB_ = {this->b, this->k, this->n};
        OffsetAccessor<RowAccessor<const T>> gB = {gB_, 0, offset_n};

        static_assert(BLOCK_K % VecLoad == 0, "");
        static_assert(BLOCK_K % InnerK == 0, "");

        static_assert(SharedMemLayoutA::s1 == BLOCK_M && SharedMemLayoutA::s2 == BLOCK_K, "");
        LayoutAccessor<SharedMemLayoutA, T> sA[2] = {this->smem, this->smem + BLOCK_M * BLOCK_K};

        static_assert(SharedMemLayoutB::s1 == BLOCK_K && SharedMemLayoutB::s2 == BLOCK_N, "");
        LayoutAccessor<SharedMemLayoutB, T> sB[2] = {this->smem + 2 * BLOCK_M * BLOCK_K, this->smem + 2 * BLOCK_M * BLOCK_K + BLOCK_N * BLOCK_K};

        static_assert(
            GemmInstance::block_m == BLOCK_M && 
            GemmInstance::block_n == BLOCK_N && 
            GemmInstance::block_k == BLOCK_K, ""
        );

        using LdgA = LdgBlockFrag<T, BLOCK_M, BLOCK_K, Warps, VecLoad>;
        using LdgB = LdgBlockFrag<T, BLOCK_K, BLOCK_N, Warps, VecLoad>;
        LdgA ldg_a;
        LdgB ldg_b;
        GemmInstance block_gemm;
        block_gemm.fill_c(0.0f);
        ldg_a.copy_g2r(gA);
        ldg_b.copy_g2r(gB);
        ldg_a.copy_r2s(sA[0]);
        ldg_b.copy_r2s(sB[0]);
        gA.inc_offset(0, BLOCK_K);
        gB.inc_offset(BLOCK_K, 0);
        __syncthreads();

        const int k_iter = cdiv(this->k, BLOCK_K) - 1;
        for (int k = 0; k < k_iter; ++k) {
            ldg_a.copy_g2r(gA);
            ldg_b.copy_g2r(gB);
            ldg_a.copy_r2s(sA[(k + 1) % 2]);
            ldg_b.copy_r2s(sB[(k + 1) % 2]);

            block_gemm.mma(sA[k % 2], sB[k % 2]);
            gA.inc_offset(0, BLOCK_K);
            gB.inc_offset(BLOCK_K, 0);
            __syncthreads();
        }
        block_gemm.mma(sA[k_iter % 2], sB[k_iter % 2]);

        RowAccessor<T> gC_ = {this->c, this->m, this->n};
        OffsetAccessor<RowAccessor<T>> gC = {gC_, offset_m, offset_n};
        block_gemm.copy_r2g(gC);
    }
};

