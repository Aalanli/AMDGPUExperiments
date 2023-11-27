#pragma once

#include "hip_utils.hpp"
#include "warp_tiles.cpp"
#include "block_tiles.cpp"


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


template <int BLOCK_M, int BLOCK_K, int BLOCK_N, int VecLoad, int InnerK, int Warps>
struct Mfma_gemmv3_ref : BasicGemmInstance<BLOCK_M, BLOCK_K, BLOCK_N, Warps> {
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
        using GemmInstance = BlockGemmV1<BLOCK_M, BLOCK_K, BLOCK_N, ATile, BTile, CTile, Mma, Warps>;

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


template <int BLOCK_M, 
          int BLOCK_K, 
          int BLOCK_N, 
          int VecLoad, 
          int InnerK, 
          int Warps,
          typename SharedMemLayoutA,
          typename SharedMemLayoutB,
          typename GemmInstance>
struct Mfma_gemmv3 : BasicGemmInstance<BLOCK_M, BLOCK_K, BLOCK_N, Warps> {
    using BasicGemmInstance<BLOCK_M, BLOCK_K, BLOCK_N, Warps>::BasicGemmInstance;

    using super = BasicGemmInstance<BLOCK_M, BLOCK_K, BLOCK_N, Warps>;

    __device__ void run() {
        const int offset_m = blockIdx.x * BLOCK_M;
        const int offset_n = blockIdx.y * BLOCK_N;

        RowAccessor<const float> gA_ = {this->a, this->m, this->k};
        OffsetAccessor<RowAccessor<const float>> gA = {gA_, offset_m, 0};
        RowAccessor<const float> gB_ = {this->b, this->k, this->n};
        OffsetAccessor<RowAccessor<const float>> gB = {gB_, 0, offset_n};

        static_assert(BLOCK_K % VecLoad == 0, "");
        static_assert(BLOCK_K % InnerK == 0, "");

        static_assert(SharedMemLayoutA::s1 == BLOCK_M && SharedMemLayoutA::s2 == BLOCK_K, "");
        auto sA = LayoutAccessor<SharedMemLayoutA, float>(this->smem);

        static_assert(SharedMemLayoutB::s1 == BLOCK_K && SharedMemLayoutB::s2 == BLOCK_N, "");
        auto sB = LayoutAccessor<SharedMemLayoutB, float>(this->smem + BLOCK_M * BLOCK_K);

        static_assert(
            GemmInstance::block_m == BLOCK_M && 
            GemmInstance::block_n == BLOCK_N && 
            GemmInstance::block_k == BLOCK_K, ""
        );

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
            __syncthreads();
            block_gemm.mma(sA, sB);
            __syncthreads();
            gA.inc_offset(0, BLOCK_K);
            gB.inc_offset(BLOCK_K, 0);

        }
        RowAccessor<float> gC_ = {this->c, this->m, this->n};
        OffsetAccessor<RowAccessor<float>> gC = {gC_, offset_m, offset_n};
        block_gemm.copy_r2g(gC);
    }
};

template <int BLOCK_M, 
          int BLOCK_K, 
          int BLOCK_N, 
          int VecLoad, 
          int InnerK, 
          int Warps,
          typename SharedMemLayoutA,
          typename SharedMemLayoutB,
          typename GemmInstance>
struct Mfma_gemmv3_Pipeline1 : BasicGemmInstance<BLOCK_M, BLOCK_K, BLOCK_N, Warps> {
    using BasicGemmInstance<BLOCK_M, BLOCK_K, BLOCK_N, Warps>::BasicGemmInstance;

    using super = BasicGemmInstance<BLOCK_M, BLOCK_K, BLOCK_N, Warps>;

    __device__ void run() {
        const int offset_m = blockIdx.x * BLOCK_M;
        const int offset_n = blockIdx.y * BLOCK_N;

        RowAccessor<const float> gA_ = {this->a, this->m, this->k};
        OffsetAccessor<RowAccessor<const float>> gA = {gA_, offset_m, 0};
        RowAccessor<const float> gB_ = {this->b, this->k, this->n};
        OffsetAccessor<RowAccessor<const float>> gB = {gB_, 0, offset_n};

        static_assert(BLOCK_K % VecLoad == 0, "");
        static_assert(BLOCK_K % InnerK == 0, "");

        static_assert(SharedMemLayoutA::s1 == BLOCK_M && SharedMemLayoutA::s2 == BLOCK_K, "");
        auto sA = LayoutAccessor<SharedMemLayoutA, float>(this->smem);

        static_assert(SharedMemLayoutB::s1 == BLOCK_K && SharedMemLayoutB::s2 == BLOCK_N, "");
        auto sB = LayoutAccessor<SharedMemLayoutB, float>(this->smem + BLOCK_M * BLOCK_K);

        static_assert(
            GemmInstance::block_m == BLOCK_M && 
            GemmInstance::block_n == BLOCK_N && 
            GemmInstance::block_k == BLOCK_K, ""
        );

        using LdgA = LdgBlockFrag<BLOCK_M, BLOCK_K, Warps, VecLoad>;
        using LdgB = LdgBlockFrag<BLOCK_K, BLOCK_N, Warps, VecLoad>;
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
        __syncthreads();

        const int ktiles = cdiv(this->k, BLOCK_K) - 1;
        for (int k = 0; k < ktiles; ++k) {
            ldg_a.copy_g2r(gA);
            ldg_b.copy_g2r(gB);
            block_gemm.mma(sA, sB);
            __syncthreads();
            ldg_a.copy_r2s(sA);
            ldg_b.copy_r2s(sB);        
            gA.inc_offset(0, BLOCK_K);
            gB.inc_offset(BLOCK_K, 0);
        }
        block_gemm.mma(sA, sB);

        RowAccessor<float> gC_ = {this->c, this->m, this->n};
        OffsetAccessor<RowAccessor<float>> gC = {gC_, offset_m, offset_n};
        block_gemm.copy_r2g(gC);
    }
};


template <int BLOCK_M, 
          int BLOCK_K, 
          int BLOCK_N, 
          int VecLoad, 
          int InnerK, 
          int Warps,
          typename SharedMemLayoutA,
          typename SharedMemLayoutB,
          typename GemmInstance>
struct Mfma_gemmv3_Pipeline2 : BasicGemmInstance<BLOCK_M, BLOCK_K, BLOCK_N, Warps> {
    using BasicGemmInstance<BLOCK_M, BLOCK_K, BLOCK_N, Warps>::BasicGemmInstance;

    using super = BasicGemmInstance<BLOCK_M, BLOCK_K, BLOCK_N, Warps>;

    static constexpr int used_smem_bytes() {
        return 2 * (BLOCK_M + BLOCK_N) * BLOCK_K * sizeof(float);
    }

    __device__ void run() {
        const int offset_m = blockIdx.x * BLOCK_M;
        const int offset_n = blockIdx.y * BLOCK_N;

        RowAccessor<const float> gA_ = {this->a, this->m, this->k};
        OffsetAccessor<RowAccessor<const float>> gA = {gA_, offset_m, 0};
        RowAccessor<const float> gB_ = {this->b, this->k, this->n};
        OffsetAccessor<RowAccessor<const float>> gB = {gB_, 0, offset_n};

        static_assert(BLOCK_K % VecLoad == 0, "");
        static_assert(BLOCK_K % InnerK == 0, "");

        static_assert(SharedMemLayoutA::s1 == BLOCK_M && SharedMemLayoutA::s2 == BLOCK_K, "");
        LayoutAccessor<SharedMemLayoutA, float> sA[2] = {this->smem, this->smem + BLOCK_M * BLOCK_K};

        static_assert(SharedMemLayoutB::s1 == BLOCK_K && SharedMemLayoutB::s2 == BLOCK_N, "");
        LayoutAccessor<SharedMemLayoutB, float> sB[2] = {this->smem + 2 * BLOCK_M * BLOCK_K, this->smem + 2 * BLOCK_M * BLOCK_K + BLOCK_N * BLOCK_K};

        static_assert(
            GemmInstance::block_m == BLOCK_M && 
            GemmInstance::block_n == BLOCK_N && 
            GemmInstance::block_k == BLOCK_K, ""
        );

        using LdgA = LdgBlockFrag<BLOCK_M, BLOCK_K, Warps, VecLoad>;
        using LdgB = LdgBlockFrag<BLOCK_K, BLOCK_N, Warps, VecLoad>;
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

        RowAccessor<float> gC_ = {this->c, this->m, this->n};
        OffsetAccessor<RowAccessor<float>> gC = {gC_, offset_m, offset_n};
        block_gemm.copy_r2g(gC);
    }
};

