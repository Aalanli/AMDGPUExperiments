#include "utils.hpp"
#include <hip/amd_detail/amd_hip_runtime.h>



/// constexpr max
template <int A, int B>
struct Max {
    static constexpr int value = A > B ? A : B;
};

__device__ __forceinline__ float warp_reduce(float data) {
    int lane = threadIdx.x % warpSize;
    for (int i = 1; i < warpSize; i *= 2)
        data += __shfl_xor(data, i);
    return data;
}

constexpr unsigned int next_power_of_2(const unsigned int a) {
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

// 1. effect of vectorized load
// 2. effect of block-level smem pipeling (global -> reg -> smem)
// 3. effect of warp-level reg pipeling (smem -> reg)
template <int BlockM,
          int BlockK,
          int BlockN,
          int WarpM,
        //   int WarpK,
          int WarpN,
          int ThreadM,
          int ThreadK,
          int ThreadN,
          int TM,
          int TK,
          int TN>
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
    static_assert(ThreadM * ThreadK * ThreadN == warpSize, "");
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

    float* sA = smem;
    float* sB = smem + BlockM * BlockK;

    for (int kb = 0; kb < (K + BlockK - 1) / BlockK; kb++) {
        // each block computes [BlockM, BlockK] x [BlockK, BlockN]
        // load a
        {
            static_assert(nthreads >= BlockK, "");
            int thread_m = threadIdx.x / BlockK;
            int thread_k = threadIdx.x % BlockK;
            for (int m = 0; m < BlockM; m += nthreads / BlockK) {
                int coord_m = (blockIdx.x * BlockM) + thread_m + m;
                int coord_k = kb * BlockK + thread_k;
                bool inbounds = coord_m < M && coord_k < K;
                sA[thread_m * BlockK + thread_k] = inbounds ? a[coord_m * K + coord_k] : 0;
            }
        }
        // load b
        {
            static_assert(nthreads >= BlockN, "");
            int thread_k = threadIdx.x / BlockN;
            int thread_n = threadIdx.x % BlockN;
            for (int k = 0; k < BlockK; k += nthreads / BlockN) {
                int coord_k = kb * BlockK + thread_k + k;
                int coord_n = blockIdx.y * BlockN + thread_n;
                bool inbounds = coord_k < K && coord_n < N;
                sB[thread_k * BlockN + thread_n] = inbounds ? b[coord_k * N + coord_n] : 0;
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
                            regs_a[im][ik] = sA[(thread_m_offset + im) * BlockK + thread_k_offset + ik];
                        }
                    }
                    for (int in = 0; in < TN; in++) {
                        for (int ik = 0; ik < TK; ik++) {
                            regs_b[ik][in] = sB[(thread_k_offset + ik) * BlockN + thread_n_offset + in];
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
            if (inbounds) {
                c[coord_m * N + coord_n] = sC[thread_m * BlockN + thread_n];
            }
        }
    }
}


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
#define ThreadN 8
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

bool LAUNCH_NAME(float* a, float* b, float* c, int m, int k, int n) {

    hipDeviceProp_t prop;
    hipGetDeviceProperties(&prop, 0);
    int warp_size = prop.warpSize;
    int smem = prop.sharedMemPerBlock;
    int regs = prop.regsPerBlock;

    if (warp_size != ThreadK * ThreadM * ThreadN) {
        printf("platform warp_size %d != ThreadK * ThreadM * ThreadN (%d, %d, %d)\n", warp_size, ThreadK, ThreadM, ThreadN);
        return false;
    }

    dim3 grid(cdiv(m, BlockM), cdiv(n, BlockN));
    dim3 block(WarpM * WarpN * warpSize);

    int used_smem = BlockM * BlockK * sizeof(TYPE) + BlockK * BlockN * sizeof(TYPE);
    used_smem = max(used_smem, BlockM * BlockN * sizeof(TYPE));
    if (used_smem > smem) {
        printf("smem overflow: %d > %d\n", used_smem, smem);
        return false;
    }

    int used_regs = sizeof(TYPE) * (TM * TK + TK * TN + TM * TN) * WarpM * WarpN;
    if (used_regs > regs) {
        printf("regs overflow: %d > %d\n", used_regs, regs);
        return false;
    }

    auto kernel = simt_gemm_kernel<BlockM, BlockK, BlockN, WarpM, WarpN, ThreadM, ThreadK, ThreadN, TM, TK, TN>; 
    hipLaunchKernelGGL(kernel, grid, block, 0, 0, a, b, c, m, n, k);
    
    // check error
    auto error = hipGetLastError();
    if (error != hipSuccess) {
        printf("Error: %s\n", hipGetErrorString(error));
        return false;
    }

    return true;
}