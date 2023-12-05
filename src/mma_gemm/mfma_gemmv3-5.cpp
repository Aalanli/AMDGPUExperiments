#include "hip_utils.hpp"
#include "mfma_tools.cpp"

#include "launch_defs.hpp"

#ifndef _SMEM_INNER_SWIZZLE
#define _SMEM_INNER_SWIZZLE 1
#endif 


EXPORT bool LAUNCH_NAME(
    const float* A, const float* B, float* C,
    int M, int K, int N, int ver
) {
    static_assert(_BLOCK_K >= _SMEM_INNER_SWIZZLE && _BLOCK_K % _SMEM_INNER_SWIZZLE == 0, "");
    using SharedMemLayoutA = ComposeLayout<SwizzleLayout<_BLOCK_M, _BLOCK_K / _SMEM_INNER_SWIZZLE>, RowLayout<1, _SMEM_INNER_SWIZZLE>>;
    using SharedMemLayoutB = TransposeLayout<ComposeLayout<SwizzleLayout<_BLOCK_N, _BLOCK_K / _SMEM_INNER_SWIZZLE>, RowLayout<1, _SMEM_INNER_SWIZZLE>>>;
    using ATile = MFMAF32_32x32x2F32_ATile<_InnerK>;
    using BTile = MFMAF32_32x32x2F32_BTile<_InnerK>;
    using CTile = MFMAF32_32x32F32_CTile;
    if (ver == 0) {
        using GemmInstance = BlockGemmV1<_BLOCK_M, _BLOCK_K, _BLOCK_N, ATile, BTile, CTile, _Warps>;
        using Gemm = Mfma_gemmv3<float, _BLOCK_M, _BLOCK_K, _BLOCK_N, _VecLoad, _InnerK, _Warps, SharedMemLayoutA, SharedMemLayoutB, GemmInstance>;
        return run_kernel<Gemm>(A, B, C, M, K, N);
    }
    if (ver == 1) {
        // pipeline inner shared to register loop
        using GemmInstance = BlockGemmV2<_BLOCK_M, _BLOCK_K, _BLOCK_N, ATile, BTile, CTile, _Warps>;
        using Gemm = Mfma_gemmv3<float, _BLOCK_M, _BLOCK_K, _BLOCK_N, _VecLoad, _InnerK, _Warps, SharedMemLayoutA, SharedMemLayoutB, GemmInstance>;
        return run_kernel<Gemm>(A, B, C, M, K, N);
    }
    if (ver == 2) {
        // pipeline outer loop with stages 1
        using GemmInstance = BlockGemmV1<_BLOCK_M, _BLOCK_K, _BLOCK_N, ATile, BTile, CTile, _Warps>;
        using Gemm = Mfma_gemmv3_Pipeline1<float, _BLOCK_M, _BLOCK_K, _BLOCK_N, _VecLoad, _InnerK, _Warps, SharedMemLayoutA, SharedMemLayoutB, GemmInstance>;
        return run_kernel<Gemm>(A, B, C, M, K, N);
    }
    if (ver == 3) {
        // pipeline outer loop with stages 1, pipeline inner loop as well
        using GemmInstance = BlockGemmV2<_BLOCK_M, _BLOCK_K, _BLOCK_N, ATile, BTile, CTile, _Warps>;
        using Gemm = Mfma_gemmv3_Pipeline1<float, _BLOCK_M, _BLOCK_K, _BLOCK_N, _VecLoad, _InnerK, _Warps, SharedMemLayoutA, SharedMemLayoutB, GemmInstance>;
        return run_kernel<Gemm>(A, B, C, M, K, N);
    }
    if (ver == 4) {
        // pipeline outer loop with stages 1
        using GemmInstance = BlockGemmV1<_BLOCK_M, _BLOCK_K, _BLOCK_N, ATile, BTile, CTile, _Warps>;
        using Gemm = Mfma_gemmv3_Pipeline1_E<float, _BLOCK_M, _BLOCK_K, _BLOCK_N, _VecLoad, _InnerK, _Warps, SharedMemLayoutA, SharedMemLayoutB, GemmInstance>;
        return run_kernel<Gemm>(A, B, C, M, K, N);
    }

    printf("ver %d does not exist\n", ver);
    return false;
}
