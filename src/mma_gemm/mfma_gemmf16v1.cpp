#include "hip_utils.hpp"
#include "layouts.cpp"
#include "mfma_tools.cpp"

#include "launch_defs.hpp"

#ifndef _SMEM_INNER_SWIZZLE
#define _SMEM_INNER_SWIZZLE 1
#endif 


EXPORT bool LAUNCH_NAME(
    const half* A, const half* B, half* C,
    int M, int K, int N, int ver
) {
    constexpr int inner_pack = Max<_SMEM_INNER_SWIZZLE, 2>::value;
    static_assert(_BLOCK_K >= inner_pack && _BLOCK_K % inner_pack == 0, "");
    {
        using SharedMemLayoutA = ComposeLayout<SwizzleLayout<_BLOCK_M, _BLOCK_K / inner_pack>, RowLayout<1, inner_pack>>;

        // RowLayout<1, 2> is a single bank
        // ----- repeat for inner_pack / 2
        // [0 2]
        // [1 3]
        using InnerPackedLayout = ComposeLayout<RowLayout<1, inner_pack / 2>, TransposeLayout<RowLayout<1, 2>>>;
        using SharedMemLayoutB = ComposeLayout<SwizzleLayout<_BLOCK_K / 2, _BLOCK_N / (inner_pack / 2)>, InnerPackedLayout>;
        using ATile = MFMAF32_16x16x16F16_ATile<_InnerK>;
        using BTile = MFMAF32_16x16x16F16_BTile<_InnerK>;
        using CTile = MFMAF32_16x16F16_CTile;
        auto check_div = [](long t) {
            return t % _VecLoad != 0;   
        };
        if (check_div(K) || check_div(N) || check_div((long) A) || check_div((long) B)) {
            printf("at least one of args K, N, A, B not divisible by %d", _VecLoad);
            return false;
        }

        if (ver == 0) {
            using GemmInstance = BlockGemmV1<_BLOCK_M, _BLOCK_K, _BLOCK_N, ATile, BTile, CTile, _Warps>;
            using Gemm = Mfma_gemmv3<half, _BLOCK_M, _BLOCK_K, _BLOCK_N, _VecLoad, _InnerK, _Warps, SharedMemLayoutA, SharedMemLayoutB, GemmInstance>;
            return run_kernel<Gemm>(A, B, C, M, K, N);
        }
        if (ver == 1) {
            // pipeline outer loop with stages 1
            using GemmInstance = BlockGemmV1<_BLOCK_M, _BLOCK_K, _BLOCK_N, ATile, BTile, CTile, _Warps>;
            using Gemm = Mfma_gemmv3_Pipeline1_E<half, _BLOCK_M, _BLOCK_K, _BLOCK_N, _VecLoad, _InnerK, _Warps, SharedMemLayoutA, SharedMemLayoutB, GemmInstance>;
            return run_kernel<Gemm>(A, B, C, M, K, N);
        }
        if (ver == 2) {
            using SharedMemLayoutA = RowLayout<_BLOCK_M, _BLOCK_K + 1>;
            using SharedMemLayoutB = RowLayout<_BLOCK_K, _BLOCK_N + 1>;
            using GemmInstance = BlockGemmV1<_BLOCK_M, _BLOCK_K, _BLOCK_N, ATile, BTile, CTile, _Warps>;
            using Gemm = Mfma_gemmv3<half, _BLOCK_M, _BLOCK_K, _BLOCK_N, _VecLoad, _InnerK, _Warps, SharedMemLayoutA, SharedMemLayoutB, GemmInstance>;
            return run_kernel<Gemm>(A, B, C, M, K, N);
        }
    }

    printf("ver %d does not exist\n", ver);
    return false;
}
