#include "hip_utils.hpp"
#include <hip/amd_detail/amd_hip_runtime.h>

#ifndef Warp_M
#define Warp_M 1
#endif

#ifndef Warp_N
#define Warp_N 1
#endif

#ifndef REP_M
#define REP_M 1
#endif

#ifndef REP_N
#define REP_N 1
#endif

#ifndef REP_K
#define REP_K 1
#endif

// constexpr int warp_size = 64; // since this is only on amd platform anyways
constexpr int mma_m = 16;
constexpr int mma_n = 16;
constexpr int mma_k = 4;
constexpr int rep_m = REP_M;
constexpr int rep_n = REP_N;
constexpr int rep_k = REP_K;
constexpr int block_m = mma_m * rep_m * Warp_M; 
constexpr int block_n = mma_n * rep_n * Warp_N;
constexpr int block_k = mma_k * rep_k;
constexpr int nthreads = Warp_M * Warp_N * warp_size;


__global__ void mfma_f32_16x16x4f32_gemm_kernel(
    const float * __restrict__ A,
    const float * __restrict__ B,
    float * __restrict__ C,
    int M, int K, int N
) {
    using float4 = __attribute__( (__vector_size__(4 * sizeof(float)) )) float;

    float regs_a[rep_m];
    float regs_b[rep_n];
    float4 regs_c[rep_m][rep_n];

    auto mma = [&]() {
        for (int i = 0; i < rep_m; ++i) {
            for (int j = 0; j < rep_n; ++j) {
                regs_c[i][j] = __builtin_amdgcn_mfma_f32_16x16x4f32(
                    regs_a[i], regs_b[j], regs_c[i][j], 0, 0, 0);
            }
        }
    };

    const int warp_id = threadIdx.x / 64;
    const int lane = threadIdx.x % 64;
    const int warp_m = warp_id / Warp_N;
    const int warp_n = warp_id % Warp_N;

    __shared__ float sA[block_m][block_k];
    __shared__ float sB[block_k][block_n];
    
    static_assert(block_k <= nthreads && nthreads % block_k == 0, "");
    auto load_a_g2s = [&](int offset_k) {
        int row = threadIdx.x % block_k;
        int col = threadIdx.x / block_k;
        constexpr int stride_m = nthreads / block_k;
        constexpr int reps = block_m / stride_m;

        for (int i = 0; i < reps; ++i) {
            int coord_m = col + i * stride_m + blockIdx.x * block_m;
            int coord_k = row + offset_k;
            bool inbounds = coord_m < M && coord_k < K;
            sA[col + i * stride_m][row] = inbounds ? A[coord_m * K + coord_k] : 0;
        }
    };

    static_assert(block_n <= nthreads && nthreads % block_n == 0, "");
    auto load_b_g2s = [&](int offset_k) {
        int row = threadIdx.x % block_n;
        int col = threadIdx.x / block_n;
        constexpr int stride_k = nthreads / block_n;
        constexpr int reps = block_k / stride_k;

        for (int i = 0; i < reps; ++i) {
            int coord_k = col + i * stride_k + offset_k;
            int coord_n = row + blockIdx.y * block_n;
            bool inbounds = coord_k < K && coord_n < N;
            sB[col + i * stride_k][row] = inbounds ? B[coord_k * N + coord_n] : 0;
        }
    };

    auto load_a_s2r = [&](int k_idx) {
        int offset_m = lane % mma_m + warp_m * rep_m * mma_m;
        int offset_K = k_idx * mma_k + lane / mma_m;

        for (int i = 0; i < rep_m; ++i) {
            regs_a[i] = sA[offset_m + i * mma_m][offset_K];
        }
    };

    auto load_b_s2r = [&](int k_idx) {
        int offset_n = lane % mma_n + warp_n * rep_n * mma_n;
        int offset_k = k_idx * mma_k + lane / mma_n;

        for (int i = 0; i < rep_n; ++i) {
            regs_b[i] = sB[offset_k][offset_n + i * mma_n];
        }
    };

    auto store_c_r2g = [&]() {
        for (int im = 0; im < rep_m; ++im) {
            for (int in = 0; in < rep_n; ++in) {
                for (int k = 0; k < 4; ++k) {
                    int coord_m = k + (lane / 16) * 4 + im * mma_m + warp_m * rep_m * mma_m + blockIdx.x * block_m;
                    int coord_n = (lane % 16) + in * mma_n + warp_n * rep_n * mma_n + blockIdx.y * block_n;
                    bool inbounds = coord_m < M && coord_n < N;
                    if (inbounds) {
                        C[coord_m * N + coord_n] = regs_c[im][in][k];
                    }
                }
            }
        }
    };

    for (int im = 0; im < rep_m; im++) {
        for (int in = 0; in < rep_n; in++) {
            regs_c[im][in] = {0.0f, 0.0f, 0.0f, 0.0f};
        }
    }

    for (int k = 0; k < cdiv(K, block_k); ++k) {
        load_a_g2s(k * block_k);
        load_b_g2s(k * block_k);
        // if (blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == 0) {
        //     for (int i = 0; i < block_n; ++i) {
        //         for (int j = 0; j < block_k; ++j) {
        //             printf("%f ", sB[j][i]);
        //         }
        //         printf("\n");
        //     }
        // }
        __syncthreads();

        for (int i = 0; i < block_k / mma_k; ++i) {
            load_a_s2r(i);
            load_b_s2r(i);
            mma();
            // if (blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == 0) {
            //     assert(regs_a[0] == 1.0f);
            //     for (int im = 0; im < rep_m; im++) {
            //         for (int in = 0; in < rep_n; in++) {
            //             for (int k = 0; k < 4; ++k) {
            //                 printf("%f ", regs_c[im][in][k]);
            //             }
            //         }
            //     }
            //     printf("\n");
            // }
        }
        __syncthreads();
    }
    // if (blockIdx.x == 0 && blockIdx.y == 0 && warp_id == 1) {
    //     for (int i = 0; i < 64; ++i) {
    //         if (lane == i) {
    //             printf("thread %d\n", i);
    //             for (int k = 0; k < 4; ++k) {
    //                 printf("%f ", regs_c[0][0][k]);
    //             }
    //             printf("\n");
    //         }
    //         __syncthreads();

    //     }
    // }


    store_c_r2g();
}


__global__ void mfma_f32_16x16x4f32_gemm_kernelv2(
    const float * __restrict__ A,
    const float * __restrict__ B,
    float * __restrict__ C,
    int M, int K, int N
) {
    using float4 = __attribute__( (__vector_size__(4 * sizeof(float)) )) float;

    float regs_a[rep_m];
    float regs_b[rep_n];
    float4 regs_c[rep_m][rep_n];

    auto mma = [&]() {
        #pragma unroll
        for (int i = 0; i < rep_m; ++i) {
            #pragma unroll
            for (int j = 0; j < rep_n; ++j) {
                regs_c[i][j] = __builtin_amdgcn_mfma_f32_16x16x4f32(
                    regs_a[i], regs_b[j], regs_c[i][j], 0, 0, 0);
            }
        }
    };

    const int warp_id = threadIdx.x / 64;
    const int lane = threadIdx.x % 64;
    const int warp_m = warp_id / Warp_N;
    const int warp_n = warp_id % Warp_N;

    __shared__ float sA[block_m][block_k];
    __shared__ float sB[block_k][block_n];
    
    static_assert(block_k <= nthreads && nthreads % block_k == 0, "");
    auto load_a_g2s = [&](int offset_k) {
        int row = threadIdx.x % block_k;
        int col = threadIdx.x / block_k;
        constexpr int stride_m = nthreads / block_k;
        constexpr int reps = block_m / stride_m;

        #pragma unroll
        for (int i = 0; i < reps; ++i) {
            int coord_m = col + i * stride_m + blockIdx.x * block_m;
            int coord_k = row + offset_k;
            bool inbounds = coord_m < M && coord_k < K;
            int im = col + i * stride_m;
            sA[im][(row + im) % block_k] = inbounds ? A[coord_m * K + coord_k] : 0;
        }
    };

    static_assert(block_n <= nthreads && nthreads % block_n == 0, "");
    auto load_b_g2s = [&](int offset_k) {
        int row = threadIdx.x % block_n;
        int col = threadIdx.x / block_n;
        constexpr int stride_k = nthreads / block_n;
        constexpr int reps = block_k / stride_k;

        #pragma unroll
        for (int i = 0; i < reps; ++i) {
            int coord_k = col + i * stride_k + offset_k;
            int coord_n = row + blockIdx.y * block_n;
            bool inbounds = coord_k < K && coord_n < N;
            sB[col + i * stride_k][row] = inbounds ? B[coord_k * N + coord_n] : 0;
        }
    };

    auto load_a_s2r = [&](int k_idx) {
        int offset_m = lane % mma_m + warp_m * rep_m * mma_m;
        int offset_K = k_idx * mma_k + lane / mma_m;

        #pragma unroll
        for (int i = 0; i < rep_m; ++i) {
            int im = offset_m + i * mma_m;
            regs_a[i] = sA[im][(offset_K + im) % block_k];
        }
    };

    auto load_b_s2r = [&](int k_idx) {
        int offset_n = lane % mma_n + warp_n * rep_n * mma_n;
        int offset_k = k_idx * mma_k + lane / mma_n;

        #pragma unroll
        for (int i = 0; i < rep_n; ++i) {
            regs_b[i] = sB[offset_k][offset_n + i * mma_n];
        }
    };

    auto store_c_r2g = [&]() {
        #pragma unroll
        for (int im = 0; im < rep_m; ++im) {
            #pragma unroll
            for (int in = 0; in < rep_n; ++in) {
                #pragma unroll
                for (int k = 0; k < 4; ++k) {
                    int coord_m = k + (lane / 16) * 4 + im * mma_m + warp_m * rep_m * mma_m + blockIdx.x * block_m;
                    int coord_n = (lane % 16) + in * mma_n + warp_n * rep_n * mma_n + blockIdx.y * block_n;
                    bool inbounds = coord_m < M && coord_n < N;
                    if (inbounds) {
                        C[coord_m * N + coord_n] = regs_c[im][in][k];
                    }
                }
            }
        }
    };

    for (int im = 0; im < rep_m; im++) {
        for (int in = 0; in < rep_n; in++) {
            regs_c[im][in] = {0.0f, 0.0f, 0.0f, 0.0f};
        }
    }

    for (int k = 0; k < cdiv(K, block_k); ++k) {
        load_a_g2s(k * block_k);
        load_b_g2s(k * block_k);
        // if (blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == 0) {
        //     for (int i = 0; i < block_n; ++i) {
        //         for (int j = 0; j < block_k; ++j) {
        //             printf("%f ", sB[j][i]);
        //         }
        //         printf("\n");
        //     }
        // }
        __syncthreads();

        for (int i = 0; i < block_k / mma_k; ++i) {
            load_a_s2r(i);
            load_b_s2r(i);
            mma();
            // if (blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == 0) {
            //     assert(regs_a[0] == 1.0f);
            //     for (int im = 0; im < rep_m; im++) {
            //         for (int in = 0; in < rep_n; in++) {
            //             for (int k = 0; k < 4; ++k) {
            //                 printf("%f ", regs_c[im][in][k]);
            //             }
            //         }
            //     }
            //     printf("\n");
            // }
        }
        __syncthreads();
    }
    // if (blockIdx.x == 0 && blockIdx.y == 0 && warp_id == 1) {
    //     for (int i = 0; i < 64; ++i) {
    //         if (lane == i) {
    //             printf("thread %d\n", i);
    //             for (int k = 0; k < 4; ++k) {
    //                 printf("%f ", regs_c[0][0][k]);
    //             }
    //             printf("\n");
    //         }
    //         __syncthreads();

    //     }
    // }


    store_c_r2g();
}

#ifndef LAUNCH_NAME
#define LAUNCH_NAME mfma_f32_16x16x4f32_gemm
#endif

EXPORT bool LAUNCH_NAME(float* a, float* b, float* c, int m, int k, int n, int version) {
    dim3 grid(cdiv(m, block_m), cdiv(n, block_n));
    dim3 block(nthreads);

    int used_smem = block_m * block_k * sizeof(float) + block_k * block_n * sizeof(float);

    hipDeviceProp_t prop;
    HIP_ASSERT(hipGetDeviceProperties(&prop, 0));
    int smem = prop.sharedMemPerBlock;

    if (used_smem > smem) {
        printf("smem overflow: %d > %d\n", used_smem, smem);
        return false;
    }
    if (version == 0)
        hipLaunchKernelGGL(mfma_f32_16x16x4f32_gemm_kernel, grid, block, 0, 0, a, b, c, m, k, n);
    else if (version == 1)
        hipLaunchKernelGGL(mfma_f32_16x16x4f32_gemm_kernelv2, grid, block, 0, 0, a, b, c, m, k, n);
    else
        return false;

    auto error = hipGetLastError();
    if (error != hipSuccess) {
        printf("Error: %s\n", hipGetErrorString(error));
        return false;
    }

    return true;
}