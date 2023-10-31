#include "utils.hpp"
#include <__clang_hip_runtime_wrapper.h>
#include <hip/amd_detail/amd_warp_functions.h>

#ifndef LAUNCH_NAME
#define LAUNCH_NAME simt_gemm
#endif

#ifndef TYPE
#define TYPE float
#endif

#ifndef BLOCK_WARPS_K
#define BLOCK_WARPS_K 2
#endif

template <int B1, int B2, int W1, int W2, int T1, int T2>
struct BlockLayout {
    enum { nw1 = B1, nw2 = B2 };
    enum { ws1 = W1, ws2 = W2 };
    enum { t1 = T1, t2 = T2 };
    enum { sh1 = B1 * W1 * T1, sh2 = B2 * W2 * T2 };
};

struct Index {
    int nw1;  
    int nw2;
    int ws1;
    int ws2;
};

template <typename T, int Sh1, int Sh2, typename Layout>
struct BlockTensor {
    static constexpr int rep1 = Sh1 / Layout::sh1;
    static constexpr int rep2 = Sh2 / Layout::sh2;

    T data[rep1][rep2][Layout::t1][Layout::t2];

    __device__ __forceinline__ Index index() {
        Index index;
        int id = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y;
        int wid = id / warpSize;
        index.nw2 = wid % Layout::nw2;
        index.nw1 = wid / Layout::nw2;

        int tid = id % warpSize;
        index.ws2 = tid % Layout::ws2;
        index.ws1 = tid / Layout::ws2;
        return index;
    }

    /// T (*f)(int i, int j)
    template <typename F>
    __device__ __forceinline__ BlockTensor<T, Sh1, Sh2, Layout> set(F&& f) {
        BlockTensor<T, Sh1, Sh2, Layout> res;
        auto idx = res.index();
        for (int r1 = 0; r1 < res.rep1; ++r1) {
            for (int r2 = 0; r2 < res.rep2; ++r2) {
                for (int t1 = 0; t1 < Layout::t1; ++t1) {
                    for (int t2 = 0; t2 < Layout::t2; ++t2) {
                        int i = r1 * Layout::sh1 + idx.nw1 * Layout::ws1 * Layout::t1 + idx.ws1 * Layout::t1 + t1;
                        int j = r2 * Layout::sh2 + idx.nw2 * Layout::ws2 * Layout::t2 + idx.ws2 * Layout::t2 + t2;
                        res.data[r1][r2][t1][t2] = f(i % Sh1, j % Sh2);
                    }
                }
            }
        }
        return res;
    }

    /// void (*f)(int i, int j, T val)
    template <typename F>
    __device__ __forceinline__ BlockTensor<T, Sh1, Sh2, Layout> get(F&& f) {
        BlockTensor<T, Sh1, Sh2, Layout> res;
        auto idx = res.index();
        for (int r1 = 0; r1 < res.rep1; ++r1) {
            for (int r2 = 0; r2 < res.rep2; ++r2) {
                for (int t1 = 0; t1 < Layout::t1; ++t1) {
                    for (int t2 = 0; t2 < Layout::t2; ++t2) {
                        int i = r1 * Layout::sh1 + idx.nw1 * Layout::ws1 * Layout::t1 + idx.ws1 * Layout::t1 + t1;
                        int j = r2 * Layout::sh2 + idx.nw2 * Layout::ws2 * Layout::t2 + idx.ws2 * Layout::t2 + t2;
                        f(i % Sh1, j % Sh2, res.data[r1][r2][t1][t2]);
                    }
                }
            }
        }
        return res;
    }
};

template <int BlockSize, int Warps>
KERNEL(Warps * 32) test_copy_kernel(const float* __restrict__ a, float* __restrict__ b) {
    using Layout = BlockLayout<1, Warps, 1, warpSize, 1, 4>;
    using Tensor = BlockTensor<float, 1, BlockSize / 4, Layout>;
    using LayoutPacked = BlockLayout<1, Warps, 1, warpSize, 1, 1>;
    using TensorPacked = BlockTensor<float4, 1, BlockSize / 4, LayoutPacked>;
    int blx = blockIdx.x;

    TensorPacked load;
    const float4* a4 = (const float4*)(a);
    load.set([&](int i, int j) {
        return a4[j + blx * BlockSize];
    });

    Tensor store = reinterpret_cast<Tensor>(load);
    __shared__ float buf[BlockSize];
    store.get([&](int i, int j, float v) {
        buf[j] = v;
    });

    TensorPacked store2;
    store2.set([&](int i, int j) {
        int tj = j * 4;
        return make_float4(buf[tj], buf[tj + 1], buf[tj + 2], buf[tj + 3]);
    });

    store2.get([&](int i, int j, float4 v) {
        float4* b4 = (float4*)(b);
        b4[j + blx * BlockSize] = v;
    });
}

// 1. effect of vectorized load
// 2. effect of block-level smem pipeling (global -> reg -> smem)
// 3. effect of warp-level reg pipeling (smem -> reg)
template <int BlockM,
          int BlockK,
          int BlockN,
          int WarpM,
          int WarpK,
          int WarpN,
          int ThreadM,
          int ThreadK,
          int ThreadN,
          int TM,
          int TK,
          int TN>
KERNEL(WarpM * WarpK * WarpN * 32) simt_gemm(
    const float* __restrict__ a, 
    const float* __restrict__ b, 
    float* __restrict__ c,
    const int M, const int N, const int K
) {
    assert(BlockM >= WarpM * ThreadM * TM);
    assert(BlockK >= WarpK * ThreadK * TK);
    assert(BlockN >= WarpN * ThreadN * TN);
    
    constexpr int nthreads = WarpM * WarpK * WarpN * 32;
    // stage 1. A[BlockM, BlockK] x B[BlockK, BlockN]
    // stage 2. A -> [m * ]
    const int smem_elems = BlockM * BlockK + BlockK * BlockN;
    __shared__ float smem[smem_elems];
    
    float* sA = smem;
    float* sB = smem + BlockM * BlockK;

    for (int kb = 0; kb < (K + BlockK - 1) / BlockK; kb++) {
        // each block computes [BlockM, BlockK] x [BlockK, BlockN]
        // load a
        {
            assert(nthreads >= BlockK);
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
            assert(nthreads >= BlockN);
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

        // load regs a
        // each warp computes [ThreadM, ThreadK] x [ThreadK, ThreadN]
        // each thread computes [TM, TK] x [TK, TN]
        int warp_id = threadIdx.x / warpSize;
        for (int wm = 0; wm < BlockM; wm += WarpM * ThreadM * TM) {
            for (int wn = 0; wn < BlockN; wn += WarpN * ThreadN * TN) {
                for (int wk = 0; wk < BlockK; wk += WarpK * ThreadK * TK) {
                    
                    float regs_a[TM][TK];
                    float regs_b[TK][TN];
                    int warp_k_offset = (warp_id % WarpK) * ThreadK;
                    int warp_m_offset = (warp_id / WarpK) * ThreadM;
                    int warp_n_offset = (warp_id / WarpK / WarpM) * ThreadN;

                    int lane = threadIdx.x % warpSize;

                    int thread_k_offset = lane % ThreadK + warp_k_offset * TK + wk;
                    int thread_m_offset = lane / ThreadK + warp_m_offset * TM + wm;
                    


                }
            }
        }
    }

}