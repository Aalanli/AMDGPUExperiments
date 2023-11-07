#include <cstdio>
#include "utils.hpp"

#ifndef LAUNCH_NAME
#define LAUNCH_NAME simt_gemm
#define IS_EXE
#endif

#ifndef TYPE
#define TYPE float
#endif

#ifndef BlockM
#define BlockM 16
#endif

#ifndef BlockK
#define BlockK 32
#endif

#ifndef BlockN
#define BlockN 64
#endif

#ifndef WarpM
#define WarpM 2
#endif

#ifndef WarpN
#define WarpN 1
#endif

#ifndef ThreadM
#define ThreadM 2
#endif

#ifndef ThreadK
#define ThreadK 1
#endif

#ifndef ThreadN
#define ThreadN 16
#endif

#ifndef TM
#define TM 4
#endif

#ifndef TK
#define TK 1
#endif

#ifndef TN
#define TN 4
#endif


__device__ __host__ constexpr int load_factor(const int nthreads, const int min_contiguous, const int dim) {
    int max_load_factor = dim / min_contiguous;
    while (max_load_factor > 1) {
        if (nthreads % (max_load_factor * min_contiguous) == 0) {
            return max_load_factor;
        }
        max_load_factor--;
    }
    return 1;
}

__device__ __host__ constexpr int calc_min_contiguous(const int dim, const int warp_size) {
    int min_contiguous = dim % warp_size == 0 ? warp_size : warp_size / 2;
    min_contiguous = dim % min_contiguous == 0 ? min_contiguous : min_contiguous / 2;
    min_contiguous = dim % min_contiguous == 0 ? min_contiguous : min_contiguous / 2;    
    min_contiguous = dim % min_contiguous == 0 ? min_contiguous : min_contiguous / 2;    
    return min_contiguous;
}


/// this function must be hit by all threads
/// LoadF: (int, int) -> T
/// StoreF: (int, int, T) -> void
template <int NThreads, int WarpSize, typename T, int D1, int D2, typename LoadF, typename StoreF>
__device__ __forceinline__ void coalesce_mem_2d(LoadF&& load_f, StoreF&& store_f) {
    int tid = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y;
    // quarter warp should still be fine for global loads
    constexpr int min_contiguous = calc_min_contiguous(D2, WarpSize);
    static_assert(D2 % min_contiguous == 0, "last dimension is not contiguous enough");
    /// find how many threads to factor along the last dimension
    constexpr int factor = load_factor(NThreads, min_contiguous, D2);
    constexpr int factor_threads = factor * min_contiguous;
    constexpr int stride = NThreads / factor_threads;
    for (int i = 0; i < D1; i += stride) {
        for (int j = 0; j < D2; j += factor_threads) {
            int tid_contiguous = tid % factor_threads;
            int tid_factor = tid / factor_threads;
            int coord_i = i + tid_factor;
            int coord_j = j + tid_contiguous;
            if (D1 % stride == 0 && stride < D1) {
                store_f(coord_i, coord_j, load_f(coord_i, coord_j));
            } else {
                if (coord_i < D1) { // coord_j is always in bounds
                    store_f(coord_i, coord_j, load_f(coord_i, coord_j));
                }
            }
        }
    }
}

/// F is a function of type (int, int) -> T
template <int NThreads, int WarpSize, typename T, int D1, int D2, typename F>
__device__ __forceinline__ void load_smem(Tensor<T, D1, D2> &a, F&& f) {
    coalesce_mem_2d<NThreads, WarpSize, T, D1, D2>(f, [&](int i, int j, T v) {
        a[i][j] = v;
    });
}

/// F is a function of type (int, int) -> T
template <int NThreads, int WarpSize, typename T, int D1, int D2, typename F>
__device__ __forceinline__ void store_gmem(Tensor<T, D1, D2> &a, F&& f) {
    coalesce_mem_2d<NThreads, WarpSize, T, D1, D2>([&](int i, int j) {
        return a[i][j];
    }, f);
}


template <int M, int K, int N>
__device__ __forceinline__ void mma(float (&a)[M][K], float (&b)[K][N], float (&c)[M][N]) {
    #pragma unroll
    for (int m = 0; m < M; ++m) {
        #pragma unroll
        for (int n = 0; n < N; ++n) {
            #pragma unroll
            for (int k = 0; k < K; ++k) {
                c[m][n] += a[m][k] * b[k][n];
            }
        }
    }
}

template <int warpSize>
__device__ __forceinline__ float warp_reduce(float data) {
    int lane = threadIdx.x % warpSize;
    for (int i = 1; i < warpSize; i *= 2)
        data += __shfl_xor(data, i);
    return data;
}


template <int D1, int D2>
__device__ void debug_print_smem(Tensor<float, D1, D2> &a) {
    if (blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == 0) {
        for (int i = 0; i < D1; ++i) {
            for (int j = 0; j < D2; ++j) {
                printf("%f ", a[i][j]);
            }
            printf("\n");
        }
        printf("\n");
    }
}


// 1. effect of vectorized load
// 2. effect of block-level smem pipeling (global -> reg -> smem)
// 3. effect of warp-level reg pipeling (smem -> reg)
template <int warpSize>
__global__ void __launch_bounds__(WarpM * /*WarpK* */ WarpN * warpSize) simt_gemm_kernelv3(
    const float* __restrict__ a, 
    const float* __restrict__ b, 
    float* __restrict__ c,
    const int M, const int N, const int K
) {
    constexpr int WarpK = 1; // if WarpK > 1, then we have to reduce in smem

    constexpr int nthreads = WarpM * WarpK * WarpN * warpSize;
    // stage 1. each block handles A[BlockM, BlockK] x B[BlockK, BlockN]
    // stage 2. each warp handles A[ThreadM, BlockK / WarpK] x B[BlockK / WarpK, ThreadN]
    // stage 3. each thread handles A[TM, TK] x B[TK, TN]
    constexpr int smem_elems = Max<2 * (BlockM * (BlockK + 1) + BlockK * (BlockN)), (BlockN + 1) * BlockM>::value;
    __shared__ float smem[smem_elems];

    static_assert(BlockK >= WarpK * ThreadK * TK,"");
    
    static_assert(BlockM >= WarpM * ThreadM * TM && BlockM % (WarpM * ThreadM * TM) == 0, "");
    static_assert(BlockN >= WarpN * ThreadN * TN && BlockN % (WarpN * ThreadN * TN) == 0, "");
    static_assert(next_power_of_2(ThreadK) == ThreadK, "");
    constexpr int WM_REP = BlockM / (WarpM * ThreadM * TM);
    constexpr int WN_REP = BlockN / (WarpN * ThreadN * TN);
    float regs_c[WM_REP][WN_REP][TM][TN];
    for (int wmr = 0; wmr < WM_REP; ++wmr) {
        for (int wnr = 0; wnr < WN_REP; ++wnr) {
            for (int tm = 0; tm < TM; ++tm) {
                for (int tn = 0; tn < TN; ++tn) {
                    regs_c[wmr][wnr][tm][tn] = 0.0f;
                }
            }
        }
    }
    
    Tensor<float, 2, BlockM, BlockK + 1> sA(smem);
    Tensor<float, 2, BlockK, BlockN> sB(smem + 2 * BlockM * (BlockK + 1));
    // float* sA = smem;
    // float* sB = smem + BlockM * BlockK;
    coalesce_mem_2d<nthreads, warpSize, float, BlockM, BlockK>([&](int i, int j) {
        int coord_m = (blockIdx.x * BlockM) + i;
        int coord_k = j;
        bool inbounds = coord_m < M && coord_k < K;
        return inbounds ? a[coord_m * K + coord_k] : 0;
    }, [&](int i, int j, float v) {
        sA[0][i][j] = v;
    });

    coalesce_mem_2d<nthreads, warpSize, float, BlockK, BlockN>([&](int i, int j) {
        int coord_k = i;
        int coord_n = blockIdx.y * BlockN + j;
        bool inbounds = coord_k < K && coord_n < N;
        return inbounds ? b[coord_k * N + coord_n] : 0;
    }, [&](int i, int j, float v) {
        sB[0][i][j] = v;
    });
    __syncthreads();

    for (int kb = 0; kb < (K + BlockK - 1) / BlockK; kb++) {

        int warp_id = threadIdx.x / warpSize;
        int warp_k_offset = (warp_id % WarpK) * ThreadK;
        int warp_m_offset = ((warp_id / WarpK) % WarpM) * ThreadM;
        int warp_n_offset = ((warp_id / WarpK) / WarpM) * ThreadN;
        for (int wm = 0; wm < WM_REP; ++wm) {
            for (int wn = 0; wn < WN_REP; ++wn) {
                float regs_a[2][TM][TK];
                float regs_b[2][TK][TN];
                int lane = threadIdx.x % warpSize;
                int thread_m_offset = ((lane / ThreadK) % ThreadM + warp_m_offset + wm * WarpM * ThreadM) * TM;
                int thread_n_offset = ((lane / ThreadK / ThreadM) + warp_n_offset + wn * WarpN * ThreadN) * TN;

                int thread_k_offset = (lane % ThreadK + warp_k_offset) * TK + 0;                
                // load into registers
                for (int im = 0; im < TM; im++) {
                    for (int ik = 0; ik < TK; ik++) {
                        regs_a[0][im][ik] = sA[kb % 2][(thread_m_offset + im)][thread_k_offset + ik];
                    }
                }
                for (int in = 0; in < TN; in++) {
                    for (int ik = 0; ik < TK; ik++) {
                        regs_b[0][ik][in] = sB[kb % 2][(thread_k_offset + ik)][thread_n_offset + in];
                    }
                }

                for (int wk = 0; wk < BlockK / (WarpK * ThreadK * TK); ++wk) {
                    int thread_k_offset = (lane % ThreadK + warp_k_offset) * TK + wk * WarpK * ThreadK * TK;
                    mma(regs_a[wk % 2], regs_b[wk % 2], regs_c[wm][wn]);
                    
                    // load into registers
                    if (wk < BlockK / (WarpK * ThreadK * TK) - 1) {
                        for (int im = 0; im < TM; im++) {
                            for (int ik = 0; ik < TK; ik++) {
                                regs_a[(wk + 1) % 2][im][ik] = sA[kb % 2][(thread_m_offset + im)][thread_k_offset + WarpK * ThreadK * TK + ik];
                            }
                        }
                        for (int in = 0; in < TN; in++) {
                            for (int ik = 0; ik < TK; ik++) {
                                regs_b[(wk + 1) % 2][ik][in] = sB[kb % 2][(thread_k_offset + WarpK * ThreadK * TK + ik)][thread_n_offset + in];
                            }
                        }
                    }
                }
            }
        }
        if (kb < (K + BlockK - 1) / BlockK - 1) {
            coalesce_mem_2d<nthreads, warpSize, float, BlockM, BlockK>([&](int i, int j) {
                int coord_m = (blockIdx.x * BlockM) + i;
                int coord_k = (kb + 1) * BlockK + j;
                bool inbounds = coord_m < M && coord_k < K;
                return inbounds ? a[coord_m * K + coord_k] : 0;
            }, [&](int i, int j, float v) {
                sA[(kb + 1) % 2][i][j] = v;
            });

            coalesce_mem_2d<nthreads, warpSize, float, BlockK, BlockN>([&](int i, int j) {
                int coord_k = (kb + 1) * BlockK + i;
                int coord_n = blockIdx.y * BlockN + j;
                bool inbounds = coord_k < K && coord_n < N;
                return inbounds ? b[coord_k * N + coord_n] : 0;
            }, [&](int i, int j, float v) {
                sB[(kb + 1) % 2][i][j] = v;
            });
        }
        __syncthreads();
    }

    static_assert(ThreadK == 1, "");
    if (ThreadK > 1) {
        // we need to reduce within the warp
        constexpr int len_c = WM_REP * WN_REP * TM * TN;
        float* rc = (float*) regs_c;
        for (int i = 0; i < len_c; ++i) {
            for (int k = 1; k < ThreadK; k *= 2) {
                rc[i] += __shfl_xor(rc[i], k);
            }
        }
    }

    Tensor<float, BlockM, BlockN + 1> sC(smem);
    // float* sC = smem;

    // {
    //     load_smem<nthreads, warpSize>(sC, [](int i, int j) {
    //         return 1.0f;
    //     });
    //     __syncthreads();
    // }
    // load into sC[BlockM, BlockN]
    {
        int warp_id = threadIdx.x / warpSize;
        int warp_k_offset = (warp_id % WarpK) * ThreadK;
        int warp_m_offset = ((warp_id / WarpK) % WarpM) * ThreadM;
        int warp_n_offset = ((warp_id / WarpK) / WarpM) * ThreadN;
        for (int wm = 0; wm < WM_REP; ++wm) {
            for (int wn = 0; wn < WN_REP; ++wn) {
                int lane = threadIdx.x % warpSize;
                if (lane % ThreadK == 0) {
                    int thread_m_offset = ((lane / ThreadK) % ThreadM + warp_m_offset + wm * WarpM * ThreadM) * TM;
                    int thread_n_offset = ((lane / ThreadK / ThreadM) + warp_n_offset + wn * WarpN * ThreadN) * TN;
                    
                    for (int im = 0; im < TM; im++) {
                        for (int in = 0; in < TN; in++) {
                            sC[thread_m_offset + im][thread_n_offset + in] = regs_c[wm][wn][im][in];
                        }
                    }
                }
            }
        }
        __syncthreads();
    }

    // debug_print_smem(sC);
    // __syncthreads();

    // copy back to global
    coalesce_mem_2d<nthreads, warpSize, float, BlockM, BlockN>([&](int i, int j) {
        return sC[i][j];
    }, [&](int i, int j, float v) {
        int coord_m = (blockIdx.x * BlockM) + i;
        int coord_n = blockIdx.y * BlockN + j;
        bool inbounds = coord_m < M && coord_n < N;
        if (inbounds) {
            c[coord_m * N + coord_n] = v;
        }
    });
}


EXPORT bool LAUNCH_NAME(float* a, float* b, float* c, int m, int k, int n) {

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    int warp_size = prop.warpSize;
    int smem = prop.sharedMemPerBlock;
    int regs = prop.regsPerBlock;

    dim3 grid(cdiv(m, BlockM), cdiv(n, BlockN));
    dim3 block(WarpM * WarpN * warp_size);

    int used_smem = BlockM * BlockK * sizeof(TYPE) + BlockK * BlockN * sizeof(TYPE);
    used_smem = hmax(used_smem, BlockM * BlockN * sizeof(TYPE));
    if (used_smem > smem) {
        printf("smem overflow: %d > %d\n", used_smem, smem);
        return false;
    }

    int used_regs = sizeof(TYPE) * (TM * TK + TK * TN + TM * TN) * WarpM * WarpN;
    if (used_regs > regs) {
        printf("regs overflow: %d > %d\n", used_regs, regs);
        return false;
    }
    if (ThreadM * ThreadK * ThreadN != warp_size) {
        printf("ThreadM * ThreadK * ThreadN != warp_size, (%d, %d, %d) != %d\n", ThreadM, ThreadK, ThreadN, warp_size);
        return false;
    }

    auto kernel = simt_gemm_kernelv3<32>; 
    kernel<<<grid, block>>>(a, b, c, m, n, k);

    

    // check error
    auto error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("Error: %s\n", cudaGetErrorString(error));
        return false;
    }

    return true;
}

#ifdef IS_EXE
int main() {
    float *a, *b, *c;
    cudaMalloc(&a, 1024 * 1024 * sizeof(float));
    cudaMalloc(&b, 1024 * 1024 * sizeof(float));
    cudaMalloc(&c, 1024 * 1024 * sizeof(float));
    LAUNCH_NAME(a, b, c, 1024, 1024, 1024);
    cudaFree(a);
    cudaFree(b);
    cudaFree(c); 
}
#endif