#include "hip_utils.hpp"
#include "layouts.cpp"
#include "mfma_tools.cpp"

#include "launch_defs.hpp"
#include <__clang_hip_runtime_wrapper.h>
#include <hip/amd_detail/amd_hip_fp16.h>

#ifndef _WARP_M
#define _WARP_M 1
#endif

#ifndef _WARP_N
#define _WARP_N 1
#endif

#ifndef _SMEM_INNER_SWIZZLE
#define _SMEM_INNER_SWIZZLE 1
#endif 


template <int N>
__device__ void vec_store(const half* src, half* dst) {
    static_assert(N == 1 || N == 2 || N == 4 || N == 8, "");

    if constexpr(N == 1) {
        dst[0] = src[0];
    } else if constexpr(N == 2) {
        ((half2*)dst)[0] = ((half2*)src)[0];
    } else if constexpr(N == 4) {
        ((float2*)dst)[0] = ((float2*)src)[0];
    } else if constexpr(N == 8) {
        ((float4*)dst)[0] = ((float4*)src)[0];
    }
}


template <typename ThisThreadBlock, int BLOCK_M, int BLOCK_K, int VecLoad, int VecStore>
struct LdgA {
    const int m, k;
    const half* ptr;
    int thread_offset_m, thread_offset_k;

    static_assert(VecLoad % VecStore == 0, "");
    using ldgParams = LdgParams<BLOCK_M, BLOCK_K, ThisThreadBlock::warps, VecLoad>;
    half regs[ldgParams::rep][VecLoad];
    
    __device__ LdgA(const half* ptr, int m, int k, int offset_m) : ptr(ptr), m(m), k(k) {
        this->thread_offset_k = (ThisThreadBlock::thread_id() % ldgParams::threads_per_row) * VecLoad;
        this->thread_offset_m = offset_m + ThisThreadBlock::thread_id() / ldgParams::threads_per_row;
    }

    __device__ void copy_g2r() {
        for (int i = 0; i < ldgParams::rep; ++i) {
            bool inbounds = thread_offset_m < m && thread_offset_k < k;
            amd_buffer_load_invalid_element_set_zero<half, VecLoad>(
                ptr, 
                ((thread_offset_m + i * ldgParams::stride_col) * k + thread_offset_k), 
                inbounds, 
                m * k, 
                &regs[i][0]);
        }
        thread_offset_k += BLOCK_K;
    }

    template <typename SA>
    __device__ void copy_r2s(SA& sA) {
        int tk = ThisThreadBlock::thread_id() % ldgParams::threads_per_row;
        int tm = ThisThreadBlock::thread_id() / ldgParams::threads_per_row;

        if constexpr(ldgParams::oversub) {
            if (tm >= BLOCK_M) {
                return;
            }
        }

        for (int i = 0; i < ldgParams::rep; ++i) {
            for (int j = 0; j < VecLoad / VecStore; j++) {
                vec_store<VecStore>(
                    &regs[i][j * VecStore], 
                    sA.index(tm + i * ldgParams::stride_col, tk * VecLoad + j * VecStore)
                );
            }
        }
    }
};

template <typename ThisThreadBlock, int BLOCK_K, int BLOCK_N, int VecLoad, int VecStore>
struct LdgB {
    const int k, n;
    const half* ptr;
    int thread_offset_k, thread_offset_n;

    static_assert(VecLoad % VecStore == 0, "");
    using ldgParams = LdgParams<BLOCK_K, BLOCK_N, ThisThreadBlock::warps, VecLoad>;
    static constexpr int stride_col = 2 * ThisThreadBlock::warps * warp_size / ldgParams::threads_per_row;
    static constexpr int rep = Max<1, BLOCK_K / stride_col>::value;
    static constexpr bool oversub = stride_col > BLOCK_K;

    half regs[ldgParams::rep][VecLoad];
    
    __device__ LdgB(const half* ptr, int k, int n, int offset_n) : ptr(ptr), k(k), n(n) {
        this->thread_offset_n = offset_n + (ThisThreadBlock::thread_id() % ldgParams::threads_per_row) * VecLoad;
        this->thread_offset_k =  ThisThreadBlock::thread_id() / ldgParams::threads_per_row;
    }

    __device__ void copy_g2r() {
        for (int i = 0; i < ldgParams::rep; ++i) {
            bool inbounds = thread_offset_m < m && thread_offset_k < k;
            amd_buffer_load_invalid_element_set_zero<half, VecLoad>(
                ptr, 
                ((thread_offset_m + i * ldgParams::stride_col) * k + thread_offset_k), 
                inbounds, 
                m * k, 
                &regs[i][0]);
        }
        thread_offset_k += BLOCK_K;
    }

    template <typename SA>
    __device__ void copy_r2s(SA& sA) {
        int tk = ThisThreadBlock::thread_id() % ldgParams::threads_per_row;
        int tm = ThisThreadBlock::thread_id() / ldgParams::threads_per_row;

        if constexpr(ldgParams::oversub) {
            if (tm >= BLOCK_M) {
                return;
            }
        }

        for (int i = 0; i < ldgParams::rep; ++i) {
            for (int j = 0; j < VecLoad / VecStore; j++) {
                vec_store<VecStore>(
                    &regs[i][j * VecStore], 
                    sA.index(tm + i * ldgParams::stride_col, tk * VecLoad + j * VecStore)
                );
            }
        }
    }
};





EXPORT bool LAUNCH_NAME(
    const half* A, const half* B, half* C,
    int M, int K, int N, int ver
) {    
    using SharedMemLayoutA = RowLayout<_BLOCK_M, _BLOCK_K + 2>;
    using SharedMemLayoutB = ComposeLayout<RowLayout<_BLOCK_K / 2, _BLOCK_N + 2>, TransposeLayout<RowLayout<1, 2>>>;


    // constexpr int inner_pack = Max<_SMEM_INNER_SWIZZLE, 2>::value;
    // static_assert(_BLOCK_K >= inner_pack && _BLOCK_K % inner_pack == 0, "");
    
    // using SharedMemLayoutA = ComposeLayout<SwizzleLayout<_BLOCK_M, _BLOCK_K / inner_pack>, RowLayout<1, inner_pack>>;

    // using InnerPackedLayout = ComposeLayout<RowLayout<1, inner_pack / 2>, TransposeLayout<RowLayout<1, 2>>>;
    // using SharedMemLayoutB = ComposeLayout<SwizzleLayout<_BLOCK_K / 2, _BLOCK_N / (inner_pack / 2)>, InnerPackedLayout>;

    using ATile = MFMAF32_32x32x8F16_ATile_Packed2<_InnerK>;
    using BTile = MFMAF32_32x32x8F16_BTile_Packed2<_InnerK>;
    using CTile = MFMAF32_32x32F16_CTile;
    if (ver == 0) {
        using GemmInstance = BlockGemmV3<_BLOCK_M, _BLOCK_K, _BLOCK_N, ATile, BTile, CTile, _WARP_M, _WARP_N>;
        using Gemm = Mfma_gemmv3<half, _BLOCK_M, _BLOCK_K, _BLOCK_N, _VecLoad, _InnerK, _WARP_M *_WARP_N, SharedMemLayoutA, SharedMemLayoutB, GemmInstance>;
        return run_kernel<Gemm>(A, B, C, M, K, N);
    }
    if (ver == 1) {
        using GemmInstance = BlockGemmV3<_BLOCK_M, _BLOCK_K, _BLOCK_N, ATile, BTile, CTile, _WARP_M, _WARP_N>;
        using Gemm = Mfma_gemmv3_Pipeline1_E<half, _BLOCK_M, _BLOCK_K, _BLOCK_N, _VecLoad, _InnerK, _WARP_M *_WARP_N, SharedMemLayoutA, SharedMemLayoutB, GemmInstance>;
        return run_kernel<Gemm>(A, B, C, M, K, N);
    }
    if (ver == 2) {
        using GemmInstance = BlockGemmV3<_BLOCK_M, _BLOCK_K, _BLOCK_N, ATile, BTile, CTile, _WARP_M, _WARP_N>;
        using Gemm = Mfma_gemmv3_Ldgv2<half, _BLOCK_M, _BLOCK_K, _BLOCK_N, _VecLoad, _InnerK, _WARP_M *_WARP_N, SharedMemLayoutA, SharedMemLayoutB, GemmInstance>;
        return run_kernel<Gemm>(A, B, C, M, K, N);
    }
    if (ver == 3) {
        using GemmInstance = BlockGemmV3<_BLOCK_M, _BLOCK_K, _BLOCK_N, ATile, BTile, CTile, _WARP_M, _WARP_N>;
        using Gemm = Mfma_gemmv3_Pipeline1_E_Ldgv2<half, _BLOCK_M, _BLOCK_K, _BLOCK_N, _VecLoad, _InnerK, _WARP_M *_WARP_N, SharedMemLayoutA, SharedMemLayoutB, GemmInstance>;
        return run_kernel<Gemm>(A, B, C, M, K, N);
    }
    
    printf("ver %d does not exist\n", ver);
    return false;
}




