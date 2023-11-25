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
