#include "utils.hpp"
#include <cstdio>
#include <hip/amd_detail/amd_hip_runtime.h>



#ifndef LAUNCH_NAME
#define LAUNCH_NAME simt_gemm
#endif

#ifndef TYPE
#define TYPE float
#endif

#ifndef BlockM
#define BlockM 128
#endif

#ifndef BlockK
#define BlockK 32
#endif

#ifndef BlockN
#define BlockN 64
#endif

#ifndef WarpM
#define WarpM 4
#endif

#ifndef WarpN
#define WarpN 2
#endif

#ifndef ThreadM
#define ThreadM 8
#endif

#ifndef ThreadK
#define ThreadK 1
#endif

#ifndef ThreadN
#define ThreadN 4
#endif

#ifndef TM
#define TM 4
#endif

#ifndef TK
#define TK 4
#endif

#ifndef TN
#define TN 4
#endif

/// multidimensional static shaped tensor
template <typename T, int D1, int... dims>
struct Tensor {
    static constexpr int dim = D1;
    static constexpr int stride = Tensor<T, dims...>::dim * Tensor<T, dims...>::stride;
    T* data;
    __host__ __device__ inline Tensor(T* ptr) : data(ptr) {}
    __host__ __device__ inline Tensor<T, dims...> operator[](int i) {
        return Tensor<T, dims...>(data + i * stride);
    }
};

template <typename T, int D1>
struct Tensor<T, D1> {
    static constexpr int dim = D1;
    static constexpr int stride = 1;
    T* data;
    __host__ __device__ inline Tensor(T* ptr) : data(ptr) {}
    __host__ __device__ inline T& operator[](int i) {
        return data[i];
    }
};

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

/// this function must be hit by all threads
/// F is a function of type (int, int) -> T
template <int NThreads, int WarpSize, typename T, int D1, int D2, typename F>
__device__ __forceinline__ void load_smem(Tensor<T, D1, D2> &a, F&& f) {
    int tid = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y;
    // half warp should still be fine for global loads
    constexpr int min_contiguous = D2 % WarpSize == 0 ? WarpSize : WarpSize / 2;
    static_assert(D2 % min_contiguous == 0, "last dimension is not contiguous enough");
    /// find how many threads to factor along the last dimension
    constexpr int factor = load_factor(NThreads, min_contiguous, D2);
    constexpr int factor_threads = factor * min_contiguous;
    for (int i = 0; i < D1; i += NThreads / factor_threads) {
        for (int j = 0; j < D2; j += factor_threads) {
            int tid_contiguous = tid % factor;
            int tid_factor = tid / factor;
            int coord_i = i + tid_factor;
            int coord_j = j + tid_contiguous;
            if ((NThreads / factor_threads) % D1 == 0) {
                a[coord_i][coord_j] = f(coord_i, coord_j);        
            } else {
                if (coord_i < D1) { // coord_j is always in bounds
                    a[coord_i][coord_j] = f(coord_i, coord_j);
                }
            }
        }
    }
}

template <int M, int K, int N>
__device__ __forceinline__ void mma(float (&a)[M][K], float (&b)[K][N], float (&c)[M][N]) {
    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            for (int k = 0; k < K; ++k) {
                c[m][n] += a[m][k] * b[k][n];
            }
        }
    }
}

/// constexpr max
template <int A, int B>
struct Max {
    static constexpr int value = A > B ? A : B;
};

int my_max(int a, int b) {
    return a > b ? a : b;
}

template <int warpSize>
__device__ __forceinline__ float warp_reduce(float data) {
    int lane = threadIdx.x % warpSize;
    for (int i = 1; i < warpSize; i *= 2)
        data += __shfl_xor(data, i);
    return data;
}

__device__ constexpr unsigned int next_power_of_2(const unsigned int a) {
    unsigned int v = a;
    v--;
    v |= v >> 1;
    v |= v >> 2;
    v |= v >> 4;
    v |= v >> 8;
    v |= v >> 16;
    v += 1;
    return v;
}

__device__ void debug_print_block(float v, dim3 strides) {
    int tid = threadIdx.x;// + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y;
    int x_tile = tid % strides.x;
    int y_tile = (tid / strides.x) % strides.y;
    int z_tile = tid / (strides.x * strides.y);

    int n_threads = blockDim.x * blockDim.y * blockDim.z;
    for (int i = 0; i < n_threads; ++i) {
        int ix = i % strides.x;
        int iy = (i / strides.x) % strides.y;
        int iz = i / (strides.x * strides.y);

        if (ix == x_tile && iy == y_tile && iz == z_tile) {
            // printf("[%d %d]: ", ix, iy);
            if (ix == 0 && iy == 0) {
                printf("z=%d [%d, %d]\n", iz, strides.y, strides.x);
            }
            printf("%f ", v);
            if (ix == strides.x - 1) {
                printf("\n");
            }
            if (ix == strides.x - 1 && iy == strides.y - 1) {
                printf("----------------\n");
            }
        }
        __syncthreads();
    }
}

// 1. effect of vectorized load
// 2. effect of block-level smem pipeling (global -> reg -> smem)
// 3. effect of warp-level reg pipeling (smem -> reg)
template <int warpSize>
KERNEL(WarpM * /*WarpK* */ WarpN * warpSize) simt_gemm_kernel(
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
    constexpr int smem_elems = Max<BlockM * BlockK + BlockK * BlockN, BlockN * BlockM>::value;
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
    
    Tensor<float, BlockM, BlockK> sA(smem);
    Tensor<float, BlockK, BlockN> sB(smem + BlockM * BlockK);
    // float* sA = smem;
    // float* sB = smem + BlockM * BlockK;

    for (int kb = 0; kb < (K + BlockK - 1) / BlockK; kb++) {
        // each block computes [BlockM, BlockK] x [BlockK, BlockN]
        // load a
        load_smem<nthreads, warpSize>(sA, [&](int i, int j) {
            int coord_m = (blockIdx.x * BlockM) + i;
            int coord_k = kb * BlockK + j;
            bool inbounds = coord_m < M && coord_k < K;
            return inbounds ? a[coord_m * K + coord_k] : 0;
        });

        load_smem<nthreads, warpSize>(sB, [&](int i, int j) {
            int coord_k = kb * BlockK + i;
            int coord_n = blockIdx.y * BlockN + j;
            bool inbounds = coord_k < K && coord_n < N;
            return inbounds ? b[coord_k * N + coord_n] : 0;
        });

        // {
        //     static_assert(nthreads >= BlockK && nthreads % BlockK == 0, "");
        //     int thread_m = threadIdx.x / BlockK;
        //     int thread_k = threadIdx.x % BlockK;
        //     for (int m = 0; m < BlockM; m += nthreads / BlockK) {
        //         int coord_m = (blockIdx.x * BlockM) + thread_m + m;
        //         int coord_k = kb * BlockK + thread_k;
        //         bool inbounds = coord_m < M && coord_k < K;
        //         sA[(thread_m + m)][thread_k] = inbounds ? a[coord_m * K + coord_k] : 0;
        //     }
        // }
        // load b
        // {
        //     static_assert(nthreads >= BlockN && nthreads % BlockN == 0, "");
        //     int thread_k = threadIdx.x / BlockN;
        //     int thread_n = threadIdx.x % BlockN;
        //     for (int k = 0; k < BlockK; k += nthreads / BlockN) {
        //         int coord_k = kb * BlockK + thread_k + k;
        //         int coord_n = blockIdx.y * BlockN + thread_n;
        //         bool inbounds = coord_k < K && coord_n < N;
        //         sB[(thread_k + k)][thread_n] = inbounds ? b[coord_k * N + coord_n] : 0;
        //     }
        // }

        __syncthreads();
        if (blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == 0) {
            // print smem
            for (int i = 0; i < BlockK; i++) {
                for (int j = 0; j < BlockM; j++) {
                    printf("%f ", sA[j][i]);
                }
                printf("\n");
            }
            printf("\n");
            // print sb
            for (int i = 0; i < BlockN; i++) {
                for (int j = 0; j < BlockK; j++) {
                    printf("%f ", sB[j][i]);
                }
                printf("\n");
            }
        }


        __syncthreads();
        // now we have A[BlockM, BlockK] x B[BlockK, BlockN] in smem

        // load regs a
        // each warp computes [ThreadM, ThreadK] x [ThreadK, ThreadN]
        // each thread computes [TM, TK] x [TK, TN]
        int warp_id = threadIdx.x / warpSize;
        int warp_k_offset = (warp_id % WarpK) * ThreadK;
        int warp_m_offset = ((warp_id / WarpK) % WarpM) * ThreadM;
        int warp_n_offset = ((warp_id / WarpK) / WarpM) * ThreadN;
        for (int wm = 0; wm < BlockM; wm += WarpM * ThreadM * TM) {
            for (int wn = 0; wn < BlockN; wn += WarpN * ThreadN * TN) {
                for (int wk = 0; wk < BlockK; wk += WarpK * ThreadK * TK) {
                    
                    float regs_a[TM][TK];
                    float regs_b[TK][TN];

                    int lane = threadIdx.x % warpSize;

                    int thread_k_offset = lane % ThreadK + warp_k_offset * TK + wk;
                    int thread_m_offset = (lane / ThreadK) % ThreadM + warp_m_offset * TM + wm;
                    int thread_n_offset = (lane / ThreadK / ThreadM) + warp_n_offset * TN + wn;

                    // load into registers
                    for (int im = 0; im < TM; im++) {
                        for (int ik = 0; ik < TK; ik++) {
                            regs_a[im][ik] = sA[(thread_m_offset + im)][thread_k_offset + ik];
                        }
                    }
                    for (int in = 0; in < TN; in++) {
                        for (int ik = 0; ik < TK; ik++) {
                            regs_b[ik][in] = sB[(thread_k_offset + ik)][thread_n_offset + in];
                        }
                    }

                    // mma
                    int wmr = wm / (WarpM * ThreadM * TM);
                    int wnr = wn / (WarpN * ThreadN * TN);
                    for (int im = 0; im < TM; im++) {
                        for (int in = 0; in < TN; in++) {
                            for (int ik = 0; ik < TK; ik++) {
                                regs_c[wmr][wnr][im][in] += regs_a[im][ik] * regs_b[ik][in];
                            }
                        }
                    }
                    

                }
            }
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

    float* sC = smem;

    {
        int tid = threadIdx.x;
        while (tid < BlockM * BlockN) {
            sC[tid] = 1.0f;
            tid += nthreads;
        }
    }
    // load into sC[BlockM, BlockN]
    {
        int warp_id = threadIdx.x / warpSize;
        int warp_k_offset = (warp_id % WarpK) * ThreadK;
        int warp_m_offset = ((warp_id / WarpK) % WarpM) * ThreadM;
        int warp_n_offset = ((warp_id / WarpK) / WarpM) * ThreadN;
        for (int wm = 0; wm < BlockM; wm += WarpM * ThreadM * TM) {
            for (int wn = 0; wn < BlockN; wn += WarpN * ThreadN * TN) {
                int lane = threadIdx.x % warpSize;
                if (lane % ThreadK == 0) {
                    int thread_m_offset = (lane / ThreadK) % ThreadM + warp_m_offset * TM + wm;
                    int thread_n_offset = (lane / ThreadK / ThreadM) + warp_n_offset * TN + wn;
                    int wmr = wm / (WarpM * ThreadM * TM);
                    int wnr = wn / (WarpN * ThreadN * TN);
                    for (int im = 0; im < TM; im++) {
                        for (int in = 0; in < TN; in++) {
                            sC[(thread_m_offset + im) * BlockN + thread_n_offset + in] = regs_c[wmr][wnr][im][in];
                        }
                    }
                }
            }
        }
        __syncthreads();
    }

    // copy back to global
    {
        static_assert(nthreads >= BlockN, "");
        int thread_m = threadIdx.x / BlockN;
        int thread_n = threadIdx.x % BlockN;
        for (int m = 0; m < BlockM; m += nthreads / BlockN) {
            int coord_m = blockIdx.x * BlockM + thread_m + m;
            int coord_n = blockIdx.y * BlockN + thread_n;
            bool inbounds = coord_m < M && coord_n < N;
            // if (blockIdx.x == 0 && blockIdx.y == 0)
            //     debug_print_block(sC[thread_m * BlockN + thread_n], dim3(ThreadM, ThreadN, 1));
            if (inbounds) {

                c[coord_m * N + coord_n] = sC[thread_m * BlockN + thread_n];
            }
        }
    }
}


EXPORT bool LAUNCH_NAME(float* a, float* b, float* c, int m, int k, int n) {

    hipDeviceProp_t prop;
    hipGetDeviceProperties(&prop, 0);
    int warp_size = prop.warpSize;
    int smem = prop.sharedMemPerBlock;
    int regs = prop.regsPerBlock;

    dim3 grid(cdiv(m, BlockM), cdiv(n, BlockN));
    dim3 block(WarpM * WarpN * warp_size);

    int used_smem = BlockM * BlockK * sizeof(TYPE) + BlockK * BlockN * sizeof(TYPE);
    used_smem = my_max(used_smem, BlockM * BlockN * sizeof(TYPE));
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
    if (warp_size == 32) {
        auto kernel = simt_gemm_kernel<32>; 
        hipLaunchKernelGGL(kernel, grid, block, 0, 0, a, b, c, m, n, k);
    } else if (warp_size == 64) {
        auto kernel = simt_gemm_kernel<64>; 
        hipLaunchKernelGGL(kernel, grid, block, 0, 0, a, b, c, m, n, k);
    }
    
    // check error
    auto error = hipGetLastError();
    if (error != hipSuccess) {
        printf("Error: %s\n", hipGetErrorString(error));
        return false;
    }

    return true;
}