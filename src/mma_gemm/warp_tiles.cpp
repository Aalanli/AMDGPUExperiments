#pragma once
#include "hip_utils.hpp"
#include "layouts.cpp"
#include <hip/amd_detail/amd_hip_runtime.h>
#include <hip/hip_fp16.h>


/// warp level tiles
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
    __device__ inline void copy_s2r(SmemAcc& sA) {
        int lane = threadIdx.x % warp_size;
        for (int i = 0; i < rep_k; ++i) {
            regs[i] = *sA.index((lane % mma_m), (lane / mma_m) + i * mma_k);
        }
    }
};

/// I don't know if this type of packing can make a difference, since it seems
/// that BLOCK_K selected rarely triggers pack_size > 1
/// the pack size should also depend on the minimum contiguity of smem
///   must be used with BTilev2
template <int BLOCK_K>
struct MFMAF32_16x16F32_ATilev2: MFMAF32_16x16F32_ATile<BLOCK_K> {
    using super = MFMAF32_16x16F32_ATile<BLOCK_K>;
    template <typename SmemAcc>
    __device__ inline void copy_s2r(SmemAcc& sA) {
        int lane = threadIdx.x % warp_size;
        constexpr int pack_size = max_pack_size(super::rep_k);
        for (int i = 0; i < super::rep_k / pack_size; ++i) {
            for (int j = 0; j < pack_size; ++j) {
                this->regs[i * pack_size + j] = *sA.index((lane % super::mma_m), (lane / super::mma_m + i * super::mma_k) * pack_size + j);
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
    __device__ inline void copy_s2r(SmemAcc& sB) {
        int lane = threadIdx.x % warp_size;
        for (int i = 0; i < rep_k; ++i) {
            regs[i] = *sB.index((lane / mma_n) + i * mma_k, lane % mma_n);
        }
    }
};

template <int BLOCK_K>
struct MFMAF32_16x16F32_BTilev2 : MFMAF32_16x16F32_BTile<BLOCK_K> {
    using super = MFMAF32_16x16F32_BTile<BLOCK_K>;
    template <typename SmemAcc>
    __device__ inline void copy_s2r(SmemAcc& sB) {
        int lane = threadIdx.x % warp_size;
        constexpr int pack_size = max_pack_size(super::rep_k);
        for (int i = 0; i < super::rep_k / pack_size; ++i) {
            for (int j = 0; j < pack_size; ++j) {
                this->regs[i * pack_size + j] = *sB.index((lane / super::mma_n + i * super::mma_k) * pack_size + j, lane % super::mma_n);
            }
        }
    }
};


struct MFMAF32_16x16F32_CTile {
    float4_t regs;

    __device__ inline void fill(float v) {
        regs[0] = v;
        regs[1] = v;
        regs[2] = v;
        regs[3] = v;
    }

    template <typename GMemC>
    __device__ inline void copy_r2g(GMemC &gC) {
        int lane = threadIdx.x % warp_size;
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            auto ptr = gC.index((lane / 16) * 4 + i, lane % 16);
            if (ptr != nullptr)
                *ptr = regs[i];
        }
    }
};

template <int BLOCK_K>
struct MFMAF32_32x32x2F32_ATile {
    static constexpr int tile_m = 32;
    static constexpr int tile_k = BLOCK_K;

    static constexpr int mma_m = 32;
    static constexpr int mma_k = 2;
    static constexpr int rep_k = BLOCK_K / mma_k;
    static_assert(is_power_of_2(BLOCK_K) && BLOCK_K >= mma_k);

    float regs[rep_k];
    template <typename SmemAcc>
    __device__ inline void copy_s2r(SmemAcc& sA) {
        int lane = threadIdx.x % warp_size;
        for (int i = 0; i < rep_k; ++i) {
            regs[i] = *sA.index((lane % mma_m), (lane / mma_m) + i * mma_k);
        }
    }
};

template <int BLOCK_K>
struct MFMAF32_32x32x2F32_BTile {
    static constexpr int tile_n = 32;
    static constexpr int tile_k = BLOCK_K;

    static constexpr int mma_k = 2;
    static constexpr int mma_n = 32;
    static constexpr int rep_k = BLOCK_K / mma_k;
    static_assert(is_power_of_2(BLOCK_K) && BLOCK_K >= mma_k);

    float regs[rep_k];
    template <typename SmemAcc>
    __device__ inline void copy_s2r(SmemAcc& sB) {
        int lane = threadIdx.x % warp_size;
        for (int i = 0; i < rep_k; ++i) {
            regs[i] = *sB.index((lane / mma_n) + i * mma_k, lane % mma_n);
        }
    }
};


struct MFMAF32_32x32F32_CTile {
    float16_t regs;

    __device__ inline void fill(float v) {
        for (int i = 0; i < 16; ++i) {
            regs[i] = v;
        }
    }

    template <typename GMemC>
    __device__ inline void copy_r2g(GMemC &gC) {
        int lane = threadIdx.x % warp_size;
        repeat<4, 4>([&](int mi, int ni) {
            auto ptr = gC.index((lane / 32) * 4 + mi * 8 + ni, lane % 32);
            if (ptr != nullptr)
                *ptr = regs[mi * 4 + ni];
        });
    }
};


template <int BLOCK_K>
struct MFMAF32_16x16x16F16_ATile {
    static constexpr int tile_m = 16;
    static constexpr int tile_k = BLOCK_K;

    static constexpr int mma_m = 16;
    static constexpr int mma_k = 16;
    static constexpr int rep_k = BLOCK_K / mma_k;
    static_assert(is_power_of_2(BLOCK_K) && BLOCK_K >= mma_k);

    half4_t regs[rep_k];
    template <typename SmemAcc>
    __device__ inline void copy_s2r(SmemAcc& sA) {
        int lane = threadIdx.x % warp_size;
        // repeat<rep_k, 2>([&](int i, int j) {
        //     half *ptr = sA.index(lane % 16, (lane / 16) * 4 + i * mma_k + j * 2);
        //     half2 value = *((half2*) ptr);
        //     regs[i][j * 2] = value.x;
        //     regs[i][j * 2 + 1] = value.y;
        // });

        // do not assume that sA is contiguous for now, hopefully the compiler can vectorize
        // accesses
        repeat<rep_k, 4>([&](int i, int j) {
            half *ptr = sA.index(lane % 16, (lane / 16) * 4 + i * mma_k + j);
            regs[i][j] = *ptr;
        });
    }
};

template <int BLOCK_K>
struct MFMAF32_16x16x16F16_BTile {
    static constexpr int tile_n = 16;
    static constexpr int tile_k = BLOCK_K;

    static constexpr int mma_k = 16;
    static constexpr int mma_n = 16;
    static constexpr int rep_k = BLOCK_K / mma_k;
    static_assert(is_power_of_2(BLOCK_K) && BLOCK_K >= mma_k);

    half4_t regs[rep_k];
    template <typename SmemAcc>
    __device__ inline void copy_s2r(SmemAcc& sB) {
        int lane = threadIdx.x % warp_size;
        repeat<rep_k, 4>([&](int i, int j) {
            half* ptr = sB.index((lane / 16) * 4 + i * mma_k + j, lane % 16);
            regs[i][j] = *ptr;
        });
    }
};


struct MFMAF32_16x16F16_CTile {
    using float4 = __attribute__( (__vector_size__(4 * sizeof(float)) )) float;
    float4 regs;

    __device__ inline void fill(float v) {
        for (int i = 0; i < 4; ++i) {
            regs[i] = v;
        }
    }

    template <typename GMemC>
    __device__ inline void copy_r2g(GMemC &gC) {
        int lane = threadIdx.x % warp_size;
        repeat<4>([&](int i) {
            auto ptr = gC.index((lane / 16) * 4 + i, lane % 16);
            if (ptr != nullptr)
                *ptr = regs[i];
        });
    }
};

template <int BLOCK_K>
struct MFMAF32_32x32x8F16_ATile {
    static constexpr int tile_m = 32;
    static constexpr int tile_k = BLOCK_K;

    static constexpr int mma_m = 32;
    static constexpr int mma_k = 8;
    static constexpr int rep_k = BLOCK_K / mma_k;
    static_assert(is_power_of_2(BLOCK_K) && BLOCK_K >= mma_k);

    half4_t regs[rep_k];
    template <typename SmemAcc>
    __device__ inline void copy_s2r(SmemAcc& sA) {
        int lane = threadIdx.x % warp_size;

        repeat<rep_k, 4>([&](int i, int j) {
            half *ptr = sA.index(lane % 32, (lane / 32) * 4 + i * mma_k + j);
            regs[i][j] = *ptr;
        });
    }
};

template <int BLOCK_K>
struct MFMAF32_32x32x8F16_BTile {
    static constexpr int tile_n = 32;
    static constexpr int tile_k = BLOCK_K;

    static constexpr int mma_k = 8;
    static constexpr int mma_n = 32;
    static constexpr int rep_k = BLOCK_K / mma_k;
    static_assert(is_power_of_2(BLOCK_K) && BLOCK_K >= mma_k);

    half4_t regs[rep_k];
    template <typename SmemAcc>
    __device__ inline void copy_s2r(SmemAcc& sB) {
        int lane = threadIdx.x % warp_size;
        repeat<rep_k, 4>([&](int i, int j) {
            half* ptr = sB.index((lane / 32) * 4 + i * mma_k + j, lane % 32);
            regs[i][j] = *ptr;
        });
    }
};


struct MFMAF32_32x32F16_CTile {
    using float16 = __attribute__( (__vector_size__(16 * sizeof(float)) )) float;
    float16 regs;

    __device__ inline void fill(float v) {
        for (int i = 0; i < 16; ++i) {
            regs[i] = v;
        }
    }

    template <typename GMemC>
    __device__ inline void copy_r2g(GMemC &gC) {
        int lane = threadIdx.x % warp_size;
        repeat<4, 4>([&](int i, int j) {
            auto ptr = gC.index((lane / 32) * 4 + i * 8 + j, lane % 32);
            if (ptr != nullptr)
                *ptr = regs[i * 4 + j];
        });
    }
};

using half16 = __attribute__( (__vector_size__(16 * sizeof(_Float16)) )) _Float16;
template <int BLOCK_K>
struct WMMAF16_16x16x16F16_ATile {
    static constexpr int tile_m = 16;
    static constexpr int tile_k = BLOCK_K;

    static constexpr int mma_m = 16;
    static constexpr int mma_k = 16;
    static constexpr int rep_k = BLOCK_K / mma_k;
    static_assert(is_power_of_2(BLOCK_K) && BLOCK_K >= mma_k);

    half16 regs[rep_k];
    template <typename SmemAcc>
    __device__ inline void copy_s2r(SmemAcc& sA) {
        int lane = threadIdx.x % 16;
        repeat<rep_k, 16>([&](int i, int j) {
            regs[i][j] = *sA.index(lane, i * mma_k + j);
        });
    }
};

template <int BLOCK_K>
struct WMMAF16_16x16x16F16_BTile {
    static constexpr int tile_n = 16;
    static constexpr int tile_k = BLOCK_K;

    static constexpr int mma_k = 16;
    static constexpr int mma_n = 16;
    static constexpr int rep_k = BLOCK_K / mma_k;
    static_assert(is_power_of_2(BLOCK_K) && BLOCK_K >= mma_k);

    half16 regs[rep_k];
    template <typename SmemAcc>
    __device__ inline void copy_s2r(SmemAcc& sB) {
        int lane = threadIdx.x % warp_size;
        repeat<rep_k, 16>([&](int i, int j) {
            regs[i][j] = *sB.index(i * mma_k + j, lane);
        });
    }
};


struct WMMAF16_16x16F16_CTile {
    half16 regs;

    __device__ inline void fill(half v) {
        for (int i = 0; i < 16; ++i) {
            regs[i] = v;
        }
    }

    template <typename GMemC>
    __device__ inline void copy_r2g(GMemC &gC) {
        int lane = threadIdx.x % 16;
        repeat<8>([&](int i) {
            int r = i * 2 + threadIdx.x / 16;
            
            auto ptr = gC.index(r, lane);
            if (ptr != nullptr)
                *ptr = regs[i * 2];
        });
    }
};


template <typename TileA, typename TileB, typename TileC>
struct TileMMA {
    __device__ inline static void mma(TileA& a, TileB& b, TileC& c) {
        assert(false && "mma tile not implemented");
    }
};

template <int BLOCK_K>
struct TileMMA<MFMAF32_16x16F32_ATile<BLOCK_K>, MFMAF32_16x16F32_BTile<BLOCK_K>, MFMAF32_16x16F32_CTile> {
    __device__ inline static void mma(MFMAF32_16x16F32_ATile<BLOCK_K> &atile, MFMAF32_16x16F32_BTile<BLOCK_K> &btile, MFMAF32_16x16F32_CTile &ctile) {
        constexpr int rep_k = MFMAF32_16x16F32_ATile<BLOCK_K>::rep_k;
        for (int i = 0; i < rep_k; ++i) {
            ctile.regs = __builtin_amdgcn_mfma_f32_16x16x4f32(
                atile.regs[i], btile.regs[i], ctile.regs, 0, 0, 0
            );
        }
    }
};



template <int BLOCK_K>
struct TileMMA<MFMAF32_16x16F32_ATilev2<BLOCK_K>, MFMAF32_16x16F32_BTilev2<BLOCK_K>, MFMAF32_16x16F32_CTile> {
    __device__ inline static void mma(MFMAF32_16x16F32_ATilev2<BLOCK_K> &atile, MFMAF32_16x16F32_BTilev2<BLOCK_K> &btile, MFMAF32_16x16F32_CTile &ctile) {
        constexpr int rep_k = MFMAF32_16x16F32_ATilev2<BLOCK_K>::rep_k;
        for (int i = 0; i < rep_k; ++i) {
            ctile.regs = __builtin_amdgcn_mfma_f32_16x16x4f32(
                atile.regs[i], btile.regs[i], ctile.regs, 0, 0, 0
            );
        }
    }
};

template <int BLOCK_K>
struct TileMMA<MFMAF32_32x32x2F32_ATile<BLOCK_K>, MFMAF32_32x32x2F32_BTile<BLOCK_K>, MFMAF32_32x32F32_CTile> {
    __device__ inline static void mma(MFMAF32_32x32x2F32_ATile<BLOCK_K> &atile, MFMAF32_32x32x2F32_BTile<BLOCK_K> &btile, MFMAF32_32x32F32_CTile &ctile) {
        constexpr int rep_k = MFMAF32_32x32x2F32_ATile<BLOCK_K>::rep_k;
        for (int i = 0; i < rep_k; ++i) {
            ctile.regs = __builtin_amdgcn_mfma_f32_32x32x2f32(
                atile.regs[i], btile.regs[i], ctile.regs, 0, 0, 0
            );
        }
    }
};

template <int BLOCK_K>
struct TileMMA<MFMAF32_16x16x16F16_ATile<BLOCK_K>, MFMAF32_16x16x16F16_BTile<BLOCK_K>, MFMAF32_16x16F16_CTile> {
    __device__ inline static void mma(MFMAF32_16x16x16F16_ATile<BLOCK_K> &atile, MFMAF32_16x16x16F16_BTile<BLOCK_K> &btile, MFMAF32_16x16F16_CTile &ctile) {
        constexpr int rep_k = MFMAF32_16x16x16F16_ATile<BLOCK_K>::rep_k;
        for (int i = 0; i < rep_k; ++i) {
            ctile.regs = __builtin_amdgcn_mfma_f32_16x16x16f16(
                atile.regs[i], btile.regs[i], ctile.regs, 0, 0, 0
            );
        }
    }
};

template <int BLOCK_K>
struct TileMMA<MFMAF32_32x32x8F16_ATile<BLOCK_K>, MFMAF32_32x32x8F16_BTile<BLOCK_K>, MFMAF32_32x32F16_CTile> {
    __device__ inline static void mma(MFMAF32_32x32x8F16_ATile<BLOCK_K> &atile, MFMAF32_32x32x8F16_BTile<BLOCK_K> &btile, MFMAF32_32x32F16_CTile &ctile) {
        constexpr int rep_k = MFMAF32_32x32x8F16_ATile<BLOCK_K>::rep_k;
        for (int i = 0; i < rep_k; ++i) {
            ctile.regs = __builtin_amdgcn_mfma_f32_32x32x8f16(
                atile.regs[i], btile.regs[i], ctile.regs, 0, 0, 0
            );
        }
    }
};

template <int BLOCK_K>
struct TileMMA<WMMAF16_16x16x16F16_ATile<BLOCK_K>, WMMAF16_16x16x16F16_BTile<BLOCK_K>, WMMAF16_16x16F16_CTile> {
    __device__ inline static void mma(WMMAF16_16x16x16F16_ATile<BLOCK_K> &atile, WMMAF16_16x16x16F16_BTile<BLOCK_K> &btile, WMMAF16_16x16F16_CTile &ctile) {
        constexpr int rep_k = WMMAF16_16x16x16F16_ATile<BLOCK_K>::rep_k;
        for (int i = 0; i < rep_k; ++i) {
            ctile.regs = __builtin_amdgcn_wmma_f16_16x16x16_f16_w32(
                atile.regs[i], btile.regs[i], ctile.regs, false
            );
        }
    }
};
