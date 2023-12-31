#include "hip_utils.hpp"
#include <hip/amd_detail/amd_hip_runtime.h>

#define DevHost __device__ __host__

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

template <int A, int B>
struct Max {
    static constexpr int value = A < B ? B : A;
};

template <int A, int B>
struct Min {
    static constexpr int value = A < B ? A : B;
};


template <typename GemmInstance>
__global__ __launch_bounds__(GemmInstance::nthreads()) void gemm_kernel(
    const float * __restrict__ A,
    const float * __restrict__ B,
    float * __restrict__ C,
    int M, int K, int N
) {
    __shared__ char smem[GemmInstance::used_smem_bytes()];
    auto inst = GemmInstance((float*) smem, A, B, C, M, K, N);
    inst.run();
}


template <typename GemmInstance>
bool run_kernel(
    const float * __restrict__ A,
    const float * __restrict__ B,
    float * __restrict__ C,
    int M, int K, int N
) {
    hipLaunchKernelGGL(gemm_kernel<GemmInstance>, GemmInstance::blocks(M, K, N), GemmInstance::threads(), 0, 0, A, B, C, M, K, N);
    auto error = hipGetLastError();
    if (error != hipSuccess) {
        printf("Error: %s\n", hipGetErrorString(error));
        return false;
    }
    return true;
}


template <int BLOCK_M, int BLOCK_K, int BLOCK_N, int Warps>
struct GemmFmfa_f32_16x16x4_f32v1 : BasicGemmInstance<BLOCK_M, BLOCK_K, BLOCK_N, Warps> {
    using BasicGemmInstance<BLOCK_M, BLOCK_K, BLOCK_N, Warps>::BasicGemmInstance;
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

    static constexpr int used_smem_bytes() { return 0; }

    // this could be moved into sgprs, if not already
    const int warp_id = threadIdx.x / warp_size;
    const int lane = threadIdx.x % warp_size;
    const int warp_m = warp_id % warps_m;
    const int warp_n = warp_id / warps_m;
    const int offset_m = blockIdx.x * BLOCK_M + warp_m * rep_m * mma_m;
    const int offset_n = blockIdx.y * BLOCK_N + warp_n * rep_n * mma_n;
    static_assert(BLOCK_K % mma_k == 0, "");

    using float4 = __attribute__( (__vector_size__(4 * sizeof(float)) )) float;
    float atile[rep_m][BLOCK_K], btile[rep_m][BLOCK_K];
    float4 regs_c[rep_m][rep_n];

    __device__ void load_a_g2r(int offset_k) {
        const int wk = lane % mma_k;
        const int wm = lane / mma_k;
        for (int i = 0; i < rep_m; ++i) {
            for (int k = 0; k < BLOCK_K / mma_k; ++k) {
                atile[i][k] = this->a[
                    (offset_m + i * mma_m) * this->k + offset_k + k * mma_k + 
                    wk + wm * this->k
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
                    (offset_k + k * mma_k) * this->n + i * mma_n + offset_n +
                    wk * this->n + wn
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
                        (offset_m + im * mma_m + lane / mma_n + k) * this->n + 
                        (offset_n + in * mma_n) + lane % mma_n
                    ] = regs_c[im][in][k];
                }
            }

        }
    }

    __device__ void run() {
        for (int k = 0; k < this->k; k += BLOCK_K) {
            load_a_g2r(k);
            load_b_g2r(k);
            mma();
        }
        store_c();
    }
};