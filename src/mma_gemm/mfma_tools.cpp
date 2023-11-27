#include "hip_utils.hpp"
#include <hip/amd_detail/amd_hip_runtime.h>

#define DevHost __device__ __host__

template <int D, typename Tail>
struct SCons {
    static constexpr int head = D;
    using tail = Tail;
};

struct STail;

template <int Dim, int... Dims>
struct DimParams {
    using value_t = SCons<Dim, typename DimParams<Dims...>::value_t>;
};

template <int Dim>
struct DimParams<Dim> {
    using value_t = SCons<Dim, STail>;
};

template <typename F, typename DimInfo, typename... Indices>
DevHost void inline repeat_impl(F& f, Indices... indices) {
    if constexpr(std::is_same<DimInfo, STail>::value) {
        f(indices...);
    } else {
        static constexpr int d = DimInfo::head;
        #pragma unroll
        for (int i = 0; i < d; ++i) {
            repeat_impl<F, typename DimInfo::tail>(f, indices..., i);
        }
    }
}

template <int... Dims, typename F>
DevHost void inline repeat(F&& f) {
    using DimInfo = typename DimParams<Dims...>::value_t;
    repeat_impl<F, DimInfo>(f);
}

static constexpr int warp_size = 64;


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
__global__ __launch_bounds__(GemmInstance::nthreads()) void gemm_kernel_dyn_smem(
    const float * __restrict__ A,
    const float * __restrict__ B,
    float * __restrict__ C,
    int M, int K, int N
) {
    extern __shared__ char smem[];
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

template <typename GemmInstance>
bool run_kernel_dyn_smem(
    const float * __restrict__ A,
    const float * __restrict__ B,
    float * __restrict__ C,
    int M, int K, int N
) {
    hipLaunchKernelGGL(gemm_kernel_dyn_smem<GemmInstance>, GemmInstance::blocks(M, K, N), GemmInstance::threads(), GemmInstance::used_smem_bytes(), 0, A, B, C, M, K, N);
    auto error = hipGetLastError();
    if (error != hipSuccess) {
        printf("Error: %s\n", hipGetErrorString(error));
        return false;
    }
    return true;
}


constexpr int next_power_of_2(int n) {
    n -= 1;
    n |= n >> 1;
    n |= n >> 2;
    n |= n >> 4;
    n |= n >> 8;
    n |= n >> 16;
    n += 1;
    return n;
}

constexpr bool is_power_of_2(int n) {
    return next_power_of_2(n) == n;
}

template <int S1, int S2>
struct RowLayout {
    static constexpr int s1 = S1;
    static constexpr int s2 = S2;

    static DevHost int index(int i, int j) {
        return i * S2 + j;
    }
};

template <int S1, int S2>
struct SwizzleLayout {
    static constexpr int s1 = S1;
    static constexpr int s2 = S2;

    static_assert(is_power_of_2(S2), "");
    static DevHost int index(int i, int j) {
        if constexpr(S1 <= S2) {
            return i * S2 + (i ^ j);
        } else {
            return i * S2 + ((i % S2) ^ j);
        }
    }
};

template <typename L>
struct TransposeLayout {
    static constexpr int s1 = L::s2;
    static constexpr int s2 = L::s1;

    static DevHost int index(int i, int j) {
        return L::index(j, i);
    }
};

template <typename L1, typename L2>
struct ComposeLayout {
    static constexpr int s1 = L1::s1 * L2::s1;
    static constexpr int s2 = L1::s2 * L2::s2;
    
    static DevHost int index(int i, int j) {
        return L1::index((i / L2::s1), (j / L2::s2)) * (L2::s1 * L2::s2) +
               L2::index((i % L2::s1), (j % L2::s2));
    }
};

template <typename L, typename T>
struct LayoutAccessor {
    using AccT = T;
    T* ptr;
    static constexpr int s1 = L::s1;
    static constexpr int s2 = L::s2;

    DevHost LayoutAccessor(T* ptr) : ptr(ptr) {}

    DevHost T* index(int i, int j) {
        return ptr + L::index(i, j);
    }
};

template <typename A>
__device__ void print_smem(A& a) {
    if (threadIdx.x == 0) {
        repeat<A::s1, A::s2>([&](int i, int j) {
            printf("%f ", (float) *a.index(i, j));
            if (j == A::s2 - 1)
                printf("\n");
        });
    }
}

template <int pack>
__device__ void vec_load(const float* ptr, float* dest) {
    static_assert(pack == 1 || pack == 2 || pack == 4, "illegal vector load number");
    if constexpr(pack == 1) {
        *dest = *ptr;
    } else if constexpr (pack == 2) {
        // well hopefully this will be optimized away if dest points to registers
        float2 v = *((float2*) ptr);
        dest[0] = v.x;
        dest[1] = v.y;
    } else {
        float4 v = *((float4*) ptr);
        dest[0] = v.x;
        dest[1] = v.y;
        dest[2] = v.z;
        dest[3] = v.w;
    }
}

template <int pack>
__device__ void vec_load_nullable(const float* ptr, float* dest) {
    static_assert(pack == 1 || pack == 2 || pack == 4, "illegal vector load number");
    if constexpr(pack == 1) {
        if (ptr)
            *dest = *ptr;
        else
            *dest = 0.0f;
    } else if constexpr (pack == 2) {
        // well hopefully this will be optimized away if dest points to registers
        float2 v;
        if (ptr)
            v = *((float2*) ptr);
        else
            v = {0.0f, 0.0f};
        dest[0] = v.x;
        dest[1] = v.y;
    } else {
        float4 v;
        if (ptr)
            v = *((float4*) ptr);
        else
            v = {0.0f, 0.0f, 0.0f, 0.0f};
        dest[0] = v.x;
        dest[1] = v.y;
        dest[2] = v.z;
        dest[3] = v.w;
    }
}


template <typename T>
struct RowAccessor {
    using AccT = T;
    T* ptr;
    int a, b;

    // RowAccessor(T* ptr, int a, int b) : ptr(ptr), a(a), b(b) {}

    DevHost T* index(int i, int j) {
        if (i < a && j < b) {
            return ptr + (i * b + j);
        }
        return nullptr;
    }
};

template <typename L>
struct OffsetAccessor {
    using AccT = typename L::AccT;
    L inner;
    int offset_a, offset_b;

    // OffsetAccessor(L inner, int offset_a, int offset_b) : inner(inner), offset_a(offset_a), offset_b(offset_b) {} 

    DevHost AccT* index(int i, int j) {
        return inner.index(i + offset_a, j + offset_b);
    }

    DevHost void set_offset(int a, int b) {
        offset_a = a;
        offset_b = b;
    }


    DevHost void inc_offset(int a, int b) {
        offset_a += a;
        offset_b += b;
    }
};



/// block level
template <typename T>
struct GMemTile {
    T* ptr;
    const int a;
    const int b; // hopefully the compiler can detect some of these are identical for a and b tiles (cse)
    int offset_a;
    int offset_b;

    __device__ T* index(int i, int j) {
        int coord_a = i + offset_a;
        int coord_b = j + offset_b;
        if (coord_a < a && coord_b < b) {
            return ptr + (coord_a * b + coord_b);
        } else {
            return nullptr;
        }
    }

    __device__ void inc_b(int i) { offset_b += i; }
    __device__ void inc_a(int i) { offset_a += i; }
};


constexpr int max_pack_size(const int n) {
    if (n <= 0) {
        return 1;
    } else if (n % 4 == 0) {
        return 4;
    } else if (n % 2 == 0) {
        return 2;
    }
    return 1;
}

/// block level
template <int BLOCK_A, int BLOCK_B, int Warps, int VecLoad>
struct LdgBlockFrag {
    static_assert(is_power_of_2(BLOCK_B) && BLOCK_B % VecLoad == 0, "");
    static_assert(is_power_of_2(BLOCK_A), "");
    static_assert(BLOCK_B <= Warps * warp_size, "not enough threads per row");
    static constexpr int threads_per_row = BLOCK_B / VecLoad;
    static constexpr int stride_col = Warps * warp_size / threads_per_row;
    static constexpr int rep_m = BLOCK_A / stride_col;
    static constexpr bool oversub = stride_col > BLOCK_A;
    
    float ldg_regs[Max<rep_m, 1>::value][VecLoad];

    template <typename GmemAcc>
    __device__ void copy_g2r(GmemAcc& gA) {
        int row = threadIdx.x % threads_per_row;
        int col = threadIdx.x / threads_per_row;
        if constexpr(oversub) {
            if (col < BLOCK_A) {
                auto ptr = gA.index(col, row * VecLoad);
                vec_load_nullable<VecLoad>(ptr, &ldg_regs[0][0]);
            }
        } else {
            for (int i = 0; i < rep_m; ++i) {
                auto ptr = gA.index(i * stride_col + col, row * VecLoad);
                // if is null, then fill with zeros, null represents out of bounds
                vec_load_nullable<VecLoad>(ptr, &ldg_regs[i][0]);
            }
        }
    }

    template <typename SmemAcc>
    __device__ void copy_r2s(SmemAcc& sA) {
        int row = threadIdx.x % threads_per_row;
        int col = threadIdx.x / threads_per_row;
        if constexpr(oversub) {
            if (col < BLOCK_A) {
                for (int j = 0; j < VecLoad; ++j)
                    *sA.index(col, row * VecLoad + j) = ldg_regs[0][j];
            }
        } else {
            for (int i = 0; i < rep_m; ++i)
                // assume that the compiler can detect consecutive memory accesses and vectorize them
                for (int j = 0; j < VecLoad; ++j)
                    *sA.index(i * stride_col + col, row * VecLoad + j) = ldg_regs[i][j];            
        }
    }

    __device__ void print_frag() {
        __syncthreads();
        if (threadIdx.x == 0) {
            printf("config: reps: %d, cols %d, rows %d, oversub %d, vecload %d\n", rep_m, stride_col, threads_per_row, oversub, VecLoad);
            printf("block_a %d, block_b %d\n", BLOCK_A, BLOCK_B);
        }
        repeat<rep_m, stride_col, threads_per_row>([&](int im, int sc, int tr) {
            if (threadIdx.x == sc * threads_per_row + tr)
                for (int i = 0; i < VecLoad; ++i)
                    printf("%f ", ldg_regs[im][i]);
            __syncthreads();
            if (((sc == stride_col - 1 && stride_col > 1) || (im == rep_m - 1 && rep_m > 1) || (tr == threads_per_row - 1)) && threadIdx.x == 0) {
                printf("\n");
            }
            __syncthreads();
        });
        __syncthreads();
    }
};


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
    __device__ void copy_s2r(SmemAcc& sA) {
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
    __device__ void copy_s2r(SmemAcc& sA) {
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
    __device__ void copy_s2r(SmemAcc& sB) {
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
    __device__ void copy_s2r(SmemAcc& sB) {
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
    using float4 = __attribute__( (__vector_size__(4 * sizeof(float)) )) float;
    float4 regs;

    __device__ void fill(float v) {
        regs[0] = v;
        regs[1] = v;
        regs[2] = v;
        regs[3] = v;
    }

    template <typename GMemC>
    __device__ void copy_r2g(GMemC &gC) {
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
    __device__ void copy_s2r(SmemAcc& sA) {
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
    __device__ void copy_s2r(SmemAcc& sB) {
        int lane = threadIdx.x % warp_size;
        for (int i = 0; i < rep_k; ++i) {
            regs[i] = *sB.index((lane / mma_n) + i * mma_k, lane % mma_n);
        }
    }
};


struct MFMAF32_32x32F32_CTile {
    using float16 = __attribute__( (__vector_size__(16 * sizeof(float)) )) float;
    float16 regs;

    __device__ void fill(float v) {
        for (int i = 0; i < 16; ++i) {
            regs[i] = v;
        }
    }

    template <typename GMemC>
    __device__ void copy_r2g(GMemC &gC) {
        int lane = threadIdx.x % warp_size;
        repeat<4, 4>([&](int mi, int ni) {
            auto ptr = gC.index((lane / 32) * 4 + mi * 8 + ni, lane % 32);
            if (ptr != nullptr)
                *ptr = regs[mi * 4 + ni];
        });
    }
};



template <typename TileA, typename TileB, typename TileC>
struct TileMMA {
    __device__ static void mma(TileA& a, TileB& b, TileC& c) {
        assert(false && "mma tile not implemented");
    }
};

template <int BLOCK_K>
struct TileMMA<MFMAF32_16x16F32_ATile<BLOCK_K>, MFMAF32_16x16F32_BTile<BLOCK_K>, MFMAF32_16x16F32_CTile> {
    __device__ static void mma(MFMAF32_16x16F32_ATile<BLOCK_K> &atile, MFMAF32_16x16F32_BTile<BLOCK_K> &btile, MFMAF32_16x16F32_CTile &ctile) {
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
    __device__ static void mma(MFMAF32_16x16F32_ATilev2<BLOCK_K> &atile, MFMAF32_16x16F32_ATilev2<BLOCK_K> &btile, MFMAF32_16x16F32_CTile &ctile) {
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
    __device__ static void mma(MFMAF32_32x32x2F32_ATile<BLOCK_K> &atile, MFMAF32_32x32x2F32_BTile<BLOCK_K> &btile, MFMAF32_32x32F32_CTile &ctile) {
        constexpr int rep_k = MFMAF32_32x32x2F32_ATile<BLOCK_K>::rep_k;
        for (int i = 0; i < rep_k; ++i) {
            ctile.regs = __builtin_amdgcn_mfma_f32_32x32x2f32(
                atile.regs[i], btile.regs[i], ctile.regs, 0, 0, 0
            );
        }
    }
};

/// block level
template <int BLOCK_M, 
          int BLOCK_K, 
          int BLOCK_N, 
          typename ATile, 
          typename BTile, 
          typename CTile, 
          int Warps>
struct BlockGemmBase {
    static constexpr int tile_m = ATile::tile_m;
    static constexpr int tile_k = ATile::tile_k;
    static_assert(ATile::tile_k == BTile::tile_k, "");    
    static constexpr int tile_n = BTile::tile_n;

    static constexpr int block_m = BLOCK_M;
    static constexpr int block_k = BLOCK_K;
    static constexpr int block_n = BLOCK_N;
    static constexpr int warps = Warps;
    static_assert(block_m % tile_m == 0, "");
    static_assert(block_k % tile_k == 0, "");
    static_assert(block_n % tile_n == 0, "");

    static constexpr int warps_m = Min<BLOCK_M / tile_m, Warps>::value;
    static constexpr int warps_n = Min<BLOCK_N / tile_n, Warps / warps_m>::value;

    static constexpr int rep_m = Max<BLOCK_M / (warps_m * tile_m), 1>::value;
    static constexpr int rep_n = Max<BLOCK_N / (warps_n * tile_n), 1>::value;
    static constexpr int rep_k = BLOCK_K / tile_k;
    // this could be removed if we check for oversubscription, eg: return from start
    static_assert(warps_m * warps_n == Warps, "");

    CTile Ctiles[rep_m][rep_n];

    __device__ void fill_c(float v) {
        repeat<rep_m, rep_n>([&](int i, int j) {
            Ctiles[i][j].fill(v);
        }); 
    }

    template <typename GMemC>
    __device__ void copy_r2g(GMemC &gC) {
        int warp = threadIdx.x / warp_size;
        int warp_m = warp % warps_m;
        int warp_n = warp / warps_m;
        repeat<rep_m, rep_n>([&](int i, int j) {
            OffsetAccessor<GMemC> gC_acc = {gC, warp_m * rep_m * tile_m + i * tile_m, warp_n * rep_n * tile_n + j * tile_n};
            Ctiles[i][j].copy_r2g(gC_acc);
        });
    }

};

template <int BLOCK_M, 
          int BLOCK_K, 
          int BLOCK_N, 
          typename ATile, 
          typename BTile, 
          typename CTile, 
          typename MMA,
          int Warps>
struct BlockGemmV1 : BlockGemmBase<BLOCK_M, BLOCK_K, BLOCK_N, ATile, BTile, CTile, Warps> {
    using super = BlockGemmBase<BLOCK_M, BLOCK_K, BLOCK_N, ATile, BTile, CTile, Warps>;

    ATile Atiles[super::rep_m];
    BTile Btiles[super::rep_n];

    template <typename SMemA, typename SMemB>
    __device__ void mma(SMemA &sA, SMemB &sB) {
        int warp = threadIdx.x / warp_size;
        int warp_m = warp % super::warps_m;
        int warp_n = warp / super::warps_m;

        for (int k = 0; k < super::rep_k; k++) {
            // load m
            for (int im = 0; im < super::rep_m; im++) {
                OffsetAccessor<SMemA> sA_acc = {
                    sA, 
                    warp_m * super::tile_m * super::rep_m + im * super::tile_m,
                    k * super::tile_k    
                };

                Atiles[im].copy_s2r(sA_acc);
            }

            // load n
            for (int in = 0; in < super::rep_n; in++) {
                OffsetAccessor<SMemB> sB_acc = {
                    sB,
                    k * super::tile_k,
                    warp_n * super::tile_n * super::rep_n + in * super::tile_n
                };

                Btiles[in].copy_s2r(sB_acc);
            }

            // mma
            repeat<super::rep_m, super::rep_n>([&](int i, int j) {
                MMA::mma(Atiles[i], Btiles[j], this->Ctiles[i][j]);
            });
        }
    }
};

/// pipeline loading from smem
template <int BLOCK_M, 
          int BLOCK_K, 
          int BLOCK_N, 
          typename ATile, 
          typename BTile, 
          typename CTile, 
          typename MMA,
          int Warps>
struct BlockGemmV2 : BlockGemmBase<BLOCK_M, BLOCK_K, BLOCK_N, ATile, BTile, CTile, Warps> {
    using super = BlockGemmBase<BLOCK_M, BLOCK_K, BLOCK_N, ATile, BTile, CTile, Warps>;

    ATile Atiles[2][super::rep_m];
    BTile Btiles[2][super::rep_n];

    template <typename SMemA>
    __device__ void load_atile(SMemA &sA, int idx, int k_offset) {
        // lol who cares about DRY?
        int warp = threadIdx.x / warp_size;
        int warp_m = warp % super::warps_m;
        int warp_n = warp / super::warps_m;

        for (int im = 0; im < super::rep_m; im++) {
            OffsetAccessor<SMemA> sA_acc = {
                sA, 
                warp_m * super::tile_m * super::rep_m + im * super::tile_m,
                k_offset    
            };

            Atiles[idx][im].copy_s2r(sA_acc);
        }
    }

    template <typename SMemB>
    __device__ void load_btile(SMemB &sB, int idx, int k_offset) {
        int warp = threadIdx.x / warp_size;
        int warp_m = warp % super::warps_m;
        int warp_n = warp / super::warps_m;

        for (int in = 0; in < super::rep_n; in++) {
            OffsetAccessor<SMemB> sB_acc = {
                sB,
                k_offset,
                warp_n * super::tile_n * super::rep_n + in * super::tile_n
            };

            Btiles[idx][in].copy_s2r(sB_acc);
        }
    }

    template <typename SMemA, typename SMemB>
    __device__ void mma(SMemA &sA, SMemB &sB) {
        int warp = threadIdx.x / warp_size;
        int warp_m = warp % super::warps_m;
        int warp_n = warp / super::warps_m;

        load_atile(sA, 0, 0);
        load_btile(sB, 0, 0);

        #pragma unroll
        for (int k = 0; k < super::rep_k; k++) {
            if (k < super::rep_k - 1) {
                load_atile(sA, (k + 1) % 2, (k + 1) * super::tile_k);
                load_btile(sB, (k + 1) % 2, (k + 1) * super::tile_k);
            }
            repeat<super::rep_m, super::rep_n>([&](int i, int j) {
                MMA::mma(Atiles[k % 2][i], Btiles[k % 2][j], this->Ctiles[i][j]);
            });
        }
    }
};


template <int BLOCK_M, int BLOCK_K, int BLOCK_N, int VecLoad, int InnerK, int Warps>
struct Mfma_gemmv3_ref : BasicGemmInstance<BLOCK_M, BLOCK_K, BLOCK_N, Warps> {
    using BasicGemmInstance<BLOCK_M, BLOCK_K, BLOCK_N, Warps>::BasicGemmInstance;

    using super = BasicGemmInstance<BLOCK_M, BLOCK_K, BLOCK_N, Warps>;

    __device__ void run() {
        const int offset_m = blockIdx.x * BLOCK_M;
        const int offset_n = blockIdx.y * BLOCK_N;


        // auto gA = OffsetAccessor<RowAccessor<const float>>(RowAccessor<const float>(this->a, this->m, this->k), offset_m, 0);
        RowAccessor<const float> gA_ = {this->a, this->m, this->k};
        OffsetAccessor<RowAccessor<const float>> gA = {gA_, offset_m, 0};
        RowAccessor<const float> gB_ = {this->b, this->k, this->n};
        OffsetAccessor<RowAccessor<const float>> gB = {gB_, 0, offset_n};
        // GMemTile<const float> gB = {this->b, this->k, this->n, 0, offset_n};

        static_assert(BLOCK_K % VecLoad == 0, "");
        static_assert(BLOCK_K % InnerK == 0, "");
        using SharedMemLayoutA = ComposeLayout<SwizzleLayout<BLOCK_M, BLOCK_K / VecLoad>, RowLayout<1, VecLoad>>;
        // using SharedMemLayoutA = RowLayout<BLOCK_M, BLOCK_K>;
        auto sA = LayoutAccessor<SharedMemLayoutA, float>(this->smem);

        using SharedMemLayoutB = TransposeLayout<ComposeLayout<SwizzleLayout<BLOCK_N, BLOCK_K / VecLoad>, RowLayout<1, VecLoad>>>;
        // using SharedMemLayoutB = RowLayout<BLOCK_K, BLOCK_N>;
        auto sB = LayoutAccessor<SharedMemLayoutB, float>(this->smem + BLOCK_M * BLOCK_K);

        using ATile = MFMAF32_16x16F32_ATile<InnerK>;
        using BTile = MFMAF32_16x16F32_BTile<InnerK>;
        using CTile = MFMAF32_16x16F32_CTile;
        using Mma = TileMMA<ATile, BTile, CTile>;
        using GemmInstance = BlockGemmV1<BLOCK_M, BLOCK_K, BLOCK_N, ATile, BTile, CTile, Mma, Warps>;

        using LdgA = LdgBlockFrag<BLOCK_M, BLOCK_K, Warps, VecLoad>;
        using LdgB = LdgBlockFrag<BLOCK_K, BLOCK_N, Warps, VecLoad>;
        LdgA ldg_a;
        LdgB ldg_b;
        GemmInstance block_gemm;
        block_gemm.fill_c(0.0f);

        for (int k = 0; k < cdiv(this->k, BLOCK_K); ++k) {
            ldg_a.copy_g2r(gA);
            ldg_b.copy_g2r(gB);
            ldg_a.copy_r2s(sA);
            ldg_b.copy_r2s(sB);
            // __syncthreads();
            // ldg_a.print_frag();
            // __syncthreads();
            // print_smem(sA);
            // print_smem(sB);
            __syncthreads();
            block_gemm.mma(sA, sB);
            __syncthreads();
            gA.inc_offset(0, BLOCK_K);
            gB.inc_offset(BLOCK_K, 0);

        }
        RowAccessor<float> gC_ = {this->c, this->m, this->n};
        OffsetAccessor<RowAccessor<float>> gC = {gC_, offset_m, offset_n};
        // GMemTile<float> gC = {this->c, this->m, this->n, offset_m, offset_n};
        block_gemm.copy_r2g(gC);
    }
};


template <int BLOCK_M, 
          int BLOCK_K, 
          int BLOCK_N, 
          int VecLoad, 
          int InnerK, 
          int Warps,
          typename SharedMemLayoutA,
          typename SharedMemLayoutB,
          typename GemmInstance>
struct Mfma_gemmv3 : BasicGemmInstance<BLOCK_M, BLOCK_K, BLOCK_N, Warps> {
    using BasicGemmInstance<BLOCK_M, BLOCK_K, BLOCK_N, Warps>::BasicGemmInstance;

    using super = BasicGemmInstance<BLOCK_M, BLOCK_K, BLOCK_N, Warps>;

    __device__ void run() {
        const int offset_m = blockIdx.x * BLOCK_M;
        const int offset_n = blockIdx.y * BLOCK_N;

        RowAccessor<const float> gA_ = {this->a, this->m, this->k};
        OffsetAccessor<RowAccessor<const float>> gA = {gA_, offset_m, 0};
        RowAccessor<const float> gB_ = {this->b, this->k, this->n};
        OffsetAccessor<RowAccessor<const float>> gB = {gB_, 0, offset_n};

        static_assert(BLOCK_K % VecLoad == 0, "");
        static_assert(BLOCK_K % InnerK == 0, "");

        static_assert(SharedMemLayoutA::s1 == BLOCK_M && SharedMemLayoutA::s2 == BLOCK_K, "");
        auto sA = LayoutAccessor<SharedMemLayoutA, float>(this->smem);

        static_assert(SharedMemLayoutB::s1 == BLOCK_K && SharedMemLayoutB::s2 == BLOCK_N, "");
        auto sB = LayoutAccessor<SharedMemLayoutB, float>(this->smem + BLOCK_M * BLOCK_K);

        static_assert(
            GemmInstance::block_m == BLOCK_M && 
            GemmInstance::block_n == BLOCK_N && 
            GemmInstance::block_k == BLOCK_K, ""
        );

        using LdgA = LdgBlockFrag<BLOCK_M, BLOCK_K, Warps, VecLoad>;
        using LdgB = LdgBlockFrag<BLOCK_K, BLOCK_N, Warps, VecLoad>;
        LdgA ldg_a;
        LdgB ldg_b;
        GemmInstance block_gemm;
        block_gemm.fill_c(0.0f);

        for (int k = 0; k < cdiv(this->k, BLOCK_K); ++k) {
            ldg_a.copy_g2r(gA);
            ldg_b.copy_g2r(gB);
            ldg_a.copy_r2s(sA);
            ldg_b.copy_r2s(sB);
            __syncthreads();
            block_gemm.mma(sA, sB);
            __syncthreads();
            gA.inc_offset(0, BLOCK_K);
            gB.inc_offset(BLOCK_K, 0);

        }
        RowAccessor<float> gC_ = {this->c, this->m, this->n};
        OffsetAccessor<RowAccessor<float>> gC = {gC_, offset_m, offset_n};
        block_gemm.copy_r2g(gC);
    }
};

template <int BLOCK_M, 
          int BLOCK_K, 
          int BLOCK_N, 
          int VecLoad, 
          int InnerK, 
          int Warps,
          typename SharedMemLayoutA,
          typename SharedMemLayoutB,
          typename GemmInstance>
struct Mfma_gemmv3_Pipeline1 : BasicGemmInstance<BLOCK_M, BLOCK_K, BLOCK_N, Warps> {
    using BasicGemmInstance<BLOCK_M, BLOCK_K, BLOCK_N, Warps>::BasicGemmInstance;

    using super = BasicGemmInstance<BLOCK_M, BLOCK_K, BLOCK_N, Warps>;

    __device__ void run() {
        const int offset_m = blockIdx.x * BLOCK_M;
        const int offset_n = blockIdx.y * BLOCK_N;

        RowAccessor<const float> gA_ = {this->a, this->m, this->k};
        OffsetAccessor<RowAccessor<const float>> gA = {gA_, offset_m, 0};
        RowAccessor<const float> gB_ = {this->b, this->k, this->n};
        OffsetAccessor<RowAccessor<const float>> gB = {gB_, 0, offset_n};

        static_assert(BLOCK_K % VecLoad == 0, "");
        static_assert(BLOCK_K % InnerK == 0, "");

        static_assert(SharedMemLayoutA::s1 == BLOCK_M && SharedMemLayoutA::s2 == BLOCK_K, "");
        auto sA = LayoutAccessor<SharedMemLayoutA, float>(this->smem);

        static_assert(SharedMemLayoutB::s1 == BLOCK_K && SharedMemLayoutB::s2 == BLOCK_N, "");
        auto sB = LayoutAccessor<SharedMemLayoutB, float>(this->smem + BLOCK_M * BLOCK_K);

        static_assert(
            GemmInstance::block_m == BLOCK_M && 
            GemmInstance::block_n == BLOCK_N && 
            GemmInstance::block_k == BLOCK_K, ""
        );

        using LdgA = LdgBlockFrag<BLOCK_M, BLOCK_K, Warps, VecLoad>;
        using LdgB = LdgBlockFrag<BLOCK_K, BLOCK_N, Warps, VecLoad>;
        LdgA ldg_a;
        LdgB ldg_b;
        GemmInstance block_gemm;
        block_gemm.fill_c(0.0f);
        ldg_a.copy_g2r(gA);
        ldg_b.copy_g2r(gB);
        ldg_a.copy_r2s(sA);
        ldg_b.copy_r2s(sB);
        gA.inc_offset(0, BLOCK_K);
        gB.inc_offset(BLOCK_K, 0);
        __syncthreads();

        const int ktiles = cdiv(this->k, BLOCK_K) - 1;
        for (int k = 0; k < ktiles; ++k) {
            ldg_a.copy_g2r(gA);
            ldg_b.copy_g2r(gB);
            block_gemm.mma(sA, sB);
            __syncthreads();
            ldg_a.copy_r2s(sA);
            ldg_b.copy_r2s(sB);        
            gA.inc_offset(0, BLOCK_K);
            gB.inc_offset(BLOCK_K, 0);
        }
        block_gemm.mma(sA, sB);

        RowAccessor<float> gC_ = {this->c, this->m, this->n};
        OffsetAccessor<RowAccessor<float>> gC = {gC_, offset_m, offset_n};
        block_gemm.copy_r2g(gC);
    }
};


template <int BLOCK_M, 
          int BLOCK_K, 
          int BLOCK_N, 
          int VecLoad, 
          int InnerK, 
          int Warps,
          typename SharedMemLayoutA,
          typename SharedMemLayoutB,
          typename GemmInstance>
struct Mfma_gemmv3_Pipeline2 : BasicGemmInstance<BLOCK_M, BLOCK_K, BLOCK_N, Warps> {
    using BasicGemmInstance<BLOCK_M, BLOCK_K, BLOCK_N, Warps>::BasicGemmInstance;

    using super = BasicGemmInstance<BLOCK_M, BLOCK_K, BLOCK_N, Warps>;

    static constexpr int used_smem_bytes() {
        return 2 * (BLOCK_M + BLOCK_N) * BLOCK_K * sizeof(float);
    }

    __device__ void run() {
        const int offset_m = blockIdx.x * BLOCK_M;
        const int offset_n = blockIdx.y * BLOCK_N;

        RowAccessor<const float> gA_ = {this->a, this->m, this->k};
        OffsetAccessor<RowAccessor<const float>> gA = {gA_, offset_m, 0};
        RowAccessor<const float> gB_ = {this->b, this->k, this->n};
        OffsetAccessor<RowAccessor<const float>> gB = {gB_, 0, offset_n};

        static_assert(BLOCK_K % VecLoad == 0, "");
        static_assert(BLOCK_K % InnerK == 0, "");

        static_assert(SharedMemLayoutA::s1 == BLOCK_M && SharedMemLayoutA::s2 == BLOCK_K, "");
        LayoutAccessor<SharedMemLayoutA, float> sA[2] = {this->smem, this->smem + BLOCK_M * BLOCK_K};

        static_assert(SharedMemLayoutB::s1 == BLOCK_K && SharedMemLayoutB::s2 == BLOCK_N, "");
        LayoutAccessor<SharedMemLayoutB, float> sB[2] = {this->smem + 2 * BLOCK_M * BLOCK_K, this->smem + 2 * BLOCK_M * BLOCK_K + BLOCK_N * BLOCK_K};

        static_assert(
            GemmInstance::block_m == BLOCK_M && 
            GemmInstance::block_n == BLOCK_N && 
            GemmInstance::block_k == BLOCK_K, ""
        );

        using LdgA = LdgBlockFrag<BLOCK_M, BLOCK_K, Warps, VecLoad>;
        using LdgB = LdgBlockFrag<BLOCK_K, BLOCK_N, Warps, VecLoad>;
        LdgA ldg_a;
        LdgB ldg_b;
        GemmInstance block_gemm;
        block_gemm.fill_c(0.0f);
        ldg_a.copy_g2r(gA);
        ldg_b.copy_g2r(gB);
        ldg_a.copy_r2s(sA[0]);
        ldg_b.copy_r2s(sB[0]);
        gA.inc_offset(0, BLOCK_K);
        gB.inc_offset(BLOCK_K, 0);
        __syncthreads();

        const int k_iter = cdiv(this->k, BLOCK_K) - 1;
        for (int k = 0; k < k_iter; ++k) {
            ldg_a.copy_g2r(gA);
            ldg_b.copy_g2r(gB);
            ldg_a.copy_r2s(sA[(k + 1) % 2]);
            ldg_b.copy_r2s(sB[(k + 1) % 2]);

            block_gemm.mma(sA[k % 2], sB[k % 2]);
            gA.inc_offset(0, BLOCK_K);
            gB.inc_offset(BLOCK_K, 0);
            __syncthreads();
        }
        block_gemm.mma(sA[k_iter % 2], sB[k_iter % 2]);

        RowAccessor<float> gC_ = {this->c, this->m, this->n};
        OffsetAccessor<RowAccessor<float>> gC = {gC_, offset_m, offset_n};
        block_gemm.copy_r2g(gC);
    }
};

