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
