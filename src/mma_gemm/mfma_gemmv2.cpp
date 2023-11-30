#include "hip_utils.hpp"
#include "mfma_tools.cpp"
#include <hip/amd_detail/amd_hip_runtime.h>
#include <stdio.h>

template <int BLOCK_M, int BLOCK_K, int BLOCK_N, int Warps>
struct GemmFmfa_f32_16x16x4_f32v1 : BasicGemmInstance<float, BLOCK_M, BLOCK_K, BLOCK_N, Warps> {
    using BasicGemmInstance<float, BLOCK_M, BLOCK_K, BLOCK_N, Warps>::BasicGemmInstance;
    static constexpr int mma_m = 16;
    static constexpr int mma_n = 16;
    static constexpr int mma_k = 4;
    static_assert(BLOCK_M % mma_m == 0, "");
    static_assert(BLOCK_N % mma_n == 0, "");
    static constexpr int warps_m = Min<BLOCK_M / mma_m, Warps>::value;
    static_assert(Warps % warps_m == 0, "");
    static constexpr int warps_n = Min<BLOCK_N / mma_n, Warps / warps_m>::value;

    static constexpr int rep_m = Max<BLOCK_M / (warps_m * mma_m), 1>::value;
    static constexpr int rep_n = Max<BLOCK_N / (warps_n * mma_n), 1>::value;
    // this could be removed if we check for oversubscription, eg: return from start
    static_assert(warps_m * warps_n == Warps, "");

    static constexpr int used_smem_bytes() { return 0; /*(BLOCK_M + BLOCK_N) * BLOCK_K * sizeof(float)*/ }

    // this could be moved into sgprs, if not already
    const int warp_id = threadIdx.x / warp_size;
    const int lane = threadIdx.x % warp_size;
    const int warp_m = warp_id % warps_m;
    const int warp_n = warp_id / warps_m;
    const int offset_m = blockIdx.x * BLOCK_M + warp_m * rep_m * mma_m;
    const int offset_n = blockIdx.y * BLOCK_N + warp_n * rep_n * mma_n;
    static_assert(BLOCK_K % (mma_k) == 0, "");

    using float4 = __attribute__( (__vector_size__(4 * sizeof(float)) )) float;
    float atile[rep_m][BLOCK_K / mma_k], btile[rep_n][BLOCK_K / mma_k];
    float4 regs_c[rep_m][rep_n];

    __device__ void load_a_g2r(int offset_k) {
        const int wk = lane % mma_k;
        const int wm = lane / mma_k;
        for (int i = 0; i < rep_m; ++i) {
            for (int k = 0; k < BLOCK_K / mma_k; ++k) {
                atile[i][k] = this->a[
                    (offset_m + i * mma_m + wm) * this->k + 
                    offset_k + k * mma_k + wk
                ];
                // in thread transpose
                //  a[wm][wk] = ...
                //  a[wk][wm] = a[wm][wk]
                atile[i][k] = __shfl(atile[i][k], (lane % mma_m) * mma_k + (lane / mma_m));
            }
        }
    }
    __device__ void load_b_g2r(int offset_k) {
        const int wn = lane % mma_n;
        const int wk = lane / mma_n;
        for (int i = 0; i < rep_n; ++i) {
            for (int k = 0; k < BLOCK_K / mma_k; ++k) {
                btile[i][k] = this->b[
                    (offset_k + k * mma_k + wk) * this->n + 
                    i * mma_n + offset_n + wn
                ];
            }
        }
    }
    __device__ void fill_c(float v) {
        for (int i = 0; i < rep_m; ++i) {
            for (int j = 0; j < rep_n; ++j) {
                regs_c[i][j] = {0.0f, 0.0f, 0.0f, 0.0f};
            }
        }
    }

    __device__ void mma() {
        for (int i = 0; i < rep_m; ++i) {
            for (int j = 0; j < rep_n; ++j) {
                for (int k = 0; k < BLOCK_K / mma_k; ++k) {
                    regs_c[i][j] = __builtin_amdgcn_mfma_f32_16x16x4f32(
                        atile[i][k], btile[j][k], regs_c[i][j], 0, 0, 0
                    );
                }
            }
        }
    }

    __device__ void store_c() {
        for (int im = 0; im < rep_m; im++) {
            for (int k = 0; k < 4; ++k) {
                for (int in = 0; in < rep_n; in++) {
                    this->c[
                        (offset_m + im * mma_m + (lane / mma_n) * 4 + k) * this->n + 
                        (offset_n + in * mma_n) + lane % mma_n
                    ] = regs_c[im][in][k];
                }
            }

        }
    }

    __device__ void run() {
        fill_c(0.0f);
        for (int k = 0; k < this->k; k += BLOCK_K) {
            load_a_g2r(k);
            load_b_g2r(k);
            mma();
        }
        store_c();
    }
};


template <int BLOCK_M, int BLOCK_K, int BLOCK_N, int Warps, int VectorPackLen = 4>
struct GemmFmfa_f32_16x16x4_f32v2 : public GemmFmfa_f32_16x16x4_f32v1<BLOCK_M, BLOCK_K, BLOCK_N, Warps> {
    using GemmFmfa_f32_16x16x4_f32v1<BLOCK_M, BLOCK_K, BLOCK_N, Warps>::GemmFmfa_f32_16x16x4_f32v1;
    using super = GemmFmfa_f32_16x16x4_f32v1<BLOCK_M, BLOCK_K, BLOCK_N, Warps>;

    static_assert(BLOCK_K % (VectorPackLen * super::mma_k) == 0, "");
    static constexpr int reps_k = BLOCK_K / (VectorPackLen * super::mma_k);

    __device__ void load_a_g2r(int offset_k) {
        const int wm = this->lane % this->mma_m;
        const int wk = this->lane / this->mma_m;
        #pragma unroll
        for (int i = 0; i < this->rep_m; ++i) {
            #pragma unroll
            for (int k = 0; k < reps_k; ++k) {
                const float* ptr = this->a + 
                    (this->offset_m + i * this->mma_m + wm) * this->k + 
                    offset_k + k * this->mma_k * VectorPackLen + wk * VectorPackLen;
                vec_load<VectorPackLen>(ptr, &this->atile[i][k * VectorPackLen]);
            }
        }
    }

    __device__ void load_b_g2r(int offset_k) {
        const int wn = this->lane % this->mma_n;
        const int wk = this->lane / this->mma_n;
        #pragma unroll
        for (int i = 0; i < this->rep_n; ++i) {
            #pragma unroll
            for (int k = 0; k < reps_k; ++k) {
                #pragma unroll
                for (int vk = 0; vk < VectorPackLen; ++vk) {
                    const float* ptr = this->b + 
                        (offset_k + k * this->mma_k * VectorPackLen + wk * VectorPackLen + vk) * this->n + 
                        i * this->mma_n + this->offset_n + wn;
                    this->btile[i][k * VectorPackLen + vk] = *ptr;
                }
            }
        }
    }

    __device__ void run() {
        this->fill_c(0.0f);
        for (int k = 0; k < this->k; k += BLOCK_K) {
            load_a_g2r(k);
            load_b_g2r(k);
            this->mma();
        }
        this->store_c();
    }

    
};


#include "launch_defs.hpp"


EXPORT bool LAUNCH_NAME(
    const float* A, const float* B, float* C,
    int M, int K, int N, int ver, int pack_len
) {
    using GemmInstance = GemmFmfa_f32_16x16x4_f32v1<_BLOCK_M, _BLOCK_K, _BLOCK_N, _Warps>;
    // printf("rep_m %d, rep_n %d\n", GemmInstance::rep_m, GemmInstance::rep_n);
    // printf("warps_m %d, warps_n %d\n", GemmInstance::warps_m, GemmInstance::warps_n);
    if (ver == 0)
        return run_kernel<GemmInstance>(
            A, B, C, M, K, N
        );
    else if (ver == 1) {
        if (K % pack_len != 0)
            return false;
        if (pack_len == 1) {
            using GemmInstancev2 = GemmFmfa_f32_16x16x4_f32v2<_BLOCK_M, _BLOCK_K, _BLOCK_N, _Warps, 1>;
            return run_kernel<GemmInstancev2>(
                A, B, C, M, K, N
            );
        } else if (pack_len == 2) {
            using GemmInstancev2 = GemmFmfa_f32_16x16x4_f32v2<_BLOCK_M, _BLOCK_K, _BLOCK_N, _Warps, 2>;
            return run_kernel<GemmInstancev2>(
                A, B, C, M, K, N
            );
        } else if (pack_len == 4) {
            using GemmInstancev2 = GemmFmfa_f32_16x16x4_f32v2<_BLOCK_M, _BLOCK_K, _BLOCK_N, _Warps, 4>;
            return run_kernel<GemmInstancev2>(
                A, B, C, M, K, N
            );
        } else {
            return false;
        }
    }
    return false;
}
