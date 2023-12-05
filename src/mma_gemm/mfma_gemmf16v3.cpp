#include "hip_utils.hpp"
#include "layouts.cpp"
#include "mfma_tools.cpp"

#include "launch_defs.hpp"

#ifndef _WARP_M
#define _WARP_M 1
#endif

#ifndef _WARP_N
#define _WARP_N 1
#endif

#ifndef _SMEM_INNER_SWIZZLE
#define _SMEM_INNER_SWIZZLE 1
#endif 




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




