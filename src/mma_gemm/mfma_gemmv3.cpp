#include "hip_utils.hpp"
#include "mfma_tools.cpp"

#include "launch_defs.hpp"


EXPORT bool LAUNCH_NAME(
    const float* A, const float* B, float* C,
    int M, int K, int N, int ver
) {
    using SharedMemLayoutA = ComposeLayout<SwizzleLayout<_BLOCK_M, _BLOCK_K / _VecLoad>, RowLayout<1, _VecLoad>>;
    using SharedMemLayoutB = TransposeLayout<ComposeLayout<SwizzleLayout<_BLOCK_N, _BLOCK_K / _VecLoad>, RowLayout<1, _VecLoad>>>;
    if (ver == 0) {
        using ATile = MFMAF32_16x16F32_ATile<_InnerK>;
        using BTile = MFMAF32_16x16F32_BTile<_InnerK>;
        using CTile = MFMAF32_16x16F32_CTile;
        using Mma = TileMMA<ATile, BTile, CTile>;
        using GemmInstance = BlockGemmV1<_BLOCK_M, _BLOCK_K, _BLOCK_N, ATile, BTile, CTile, Mma, _Warps>;

        using Gemm = Mfma_gemmv3<_BLOCK_M, _BLOCK_K, _BLOCK_N, _VecLoad, _InnerK, _Warps, SharedMemLayoutA, SharedMemLayoutB, GemmInstance>;
        return run_kernel<Gemm>(A, B, C, M, K, N);
    }
    if (ver == 1) {
        // pack inner warp tile loading to perhaps vectorize loads from smem
        using ATile = MFMAF32_16x16F32_ATilev2<_InnerK>;
        using BTile = MFMAF32_16x16F32_BTilev2<_InnerK>;
        using CTile = MFMAF32_16x16F32_CTile;
        using Mma = TileMMA<ATile, BTile, CTile>;
        using GemmInstance = BlockGemmV1<_BLOCK_M, _BLOCK_K, _BLOCK_N, ATile, BTile, CTile, Mma, _Warps>;

        using Gemm = Mfma_gemmv3<_BLOCK_M, _BLOCK_K, _BLOCK_N, _VecLoad, _InnerK, _Warps, SharedMemLayoutA, SharedMemLayoutB, GemmInstance>;
        return run_kernel<Gemm>(A, B, C, M, K, N);
    }
    if (ver == 2) {
        // pipeline inner shared to register loop
        using ATile = MFMAF32_16x16F32_ATile<_InnerK>;
        using BTile = MFMAF32_16x16F32_BTile<_InnerK>;
        using CTile = MFMAF32_16x16F32_CTile;
        using Mma = TileMMA<ATile, BTile, CTile>;
        using GemmInstance = BlockGemmV2<_BLOCK_M, _BLOCK_K, _BLOCK_N, ATile, BTile, CTile, Mma, _Warps>;

        using Gemm = Mfma_gemmv3<_BLOCK_M, _BLOCK_K, _BLOCK_N, _VecLoad, _InnerK, _Warps, SharedMemLayoutA, SharedMemLayoutB, GemmInstance>;
        return run_kernel<Gemm>(A, B, C, M, K, N);
    }
    if (ver == 3) {
        // pipeline outer global to shared loop
        using ATile = MFMAF32_16x16F32_ATile<_InnerK>;
        using BTile = MFMAF32_16x16F32_BTile<_InnerK>;
        using CTile = MFMAF32_16x16F32_CTile;
        using Mma = TileMMA<ATile, BTile, CTile>;
        using GemmInstance = BlockGemmV1<_BLOCK_M, _BLOCK_K, _BLOCK_N, ATile, BTile, CTile, Mma, _Warps>;

        using Gemm = Mfma_gemmv3_Pipeline2<_BLOCK_M, _BLOCK_K, _BLOCK_N, _VecLoad, _InnerK, _Warps, SharedMemLayoutA, SharedMemLayoutB, GemmInstance>;
        return run_kernel<Gemm>(A, B, C, M, K, N);
    }
    if (ver == 4) {
        // pipeline both inner and outer loops
        using ATile = MFMAF32_16x16F32_ATile<_InnerK>;
        using BTile = MFMAF32_16x16F32_BTile<_InnerK>;
        using CTile = MFMAF32_16x16F32_CTile;
        using Mma = TileMMA<ATile, BTile, CTile>;
        using GemmInstance = BlockGemmV2<_BLOCK_M, _BLOCK_K, _BLOCK_N, ATile, BTile, CTile, Mma, _Warps>;

        using Gemm = Mfma_gemmv3_Pipeline2<_BLOCK_M, _BLOCK_K, _BLOCK_N, _VecLoad, _InnerK, _Warps, SharedMemLayoutA, SharedMemLayoutB, GemmInstance>;
        return run_kernel<Gemm>(A, B, C, M, K, N);
    }
    if (ver == 5) {
        // pipeline outer loop with stages 1
        using ATile = MFMAF32_16x16F32_ATile<_InnerK>;
        using BTile = MFMAF32_16x16F32_BTile<_InnerK>;
        using CTile = MFMAF32_16x16F32_CTile;
        using Mma = TileMMA<ATile, BTile, CTile>;
        using GemmInstance = BlockGemmV1<_BLOCK_M, _BLOCK_K, _BLOCK_N, ATile, BTile, CTile, Mma, _Warps>;

        using Gemm = Mfma_gemmv3_Pipeline1<_BLOCK_M, _BLOCK_K, _BLOCK_N, _VecLoad, _InnerK, _Warps, SharedMemLayoutA, SharedMemLayoutB, GemmInstance>;
        return run_kernel<Gemm>(A, B, C, M, K, N);
    }
    if (ver == 6) {
        using ATile = MFMAF32_16x16F32_ATile<_InnerK>;
        using BTile = MFMAF32_16x16F32_BTile<_InnerK>;
        using CTile = MFMAF32_16x16F32_CTile;
        using Mma = TileMMA<ATile, BTile, CTile>;
        using GemmInstance = BlockGemmV2<_BLOCK_M, _BLOCK_K, _BLOCK_N, ATile, BTile, CTile, Mma, _Warps>;

        using Gemm = Mfma_gemmv3_Pipeline1<_BLOCK_M, _BLOCK_K, _BLOCK_N, _VecLoad, _InnerK, _Warps, SharedMemLayoutA, SharedMemLayoutB, GemmInstance>;
        return run_kernel<Gemm>(A, B, C, M, K, N);
    }

    printf("ver %d does not exist\n", ver);
    return false;
}
