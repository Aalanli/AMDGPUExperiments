#pragma once

#include "hip/hip_runtime.h"
#include <hip/amd_detail/amd_hip_runtime.h>
#include <hip/hip_fp16.h>
// #include <hip/amd_detail/amd_hip_vector_types.h>
#include <hip/hip_runtime_api.h>

#include <stdio.h>
#include <stdlib.h>


#ifndef EXPORT
#define EXPORT extern "C" __attribute__((visibility("default")))
#endif


#define KERNEL(lb_) static __global__ __launch_bounds__((lb_)) void


#ifndef HIP_ASSERT
#define HIP_ASSERT(error)                    \
    if(error != hipSuccess)                       \
    {                                             \
        fprintf(stderr,                           \
                "hip error: '%s'(%d) at %s:%d\n", \
                hipGetErrorString(error),         \
                error,                            \
                __FILE__,                         \
                __LINE__);                        \
        exit(EXIT_FAILURE);                       \
    }
#endif


__device__ __host__ int inline cdiv(int a, int b) {
    return (a + b - 1) / b;
}


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

#ifndef __WARP_SIZE_AMDGCN__
#define __WARP_SIZE_AMDGCN__ 64
#endif

static constexpr int warp_size = __WARP_SIZE_AMDGCN__;


template <int A, int B>
struct Max {
    static constexpr int value = A < B ? B : A;
};

template <int A, int B>
struct Min {
    static constexpr int value = A < B ? A : B;
};


constexpr inline int next_power_of_2(int n) {
    n -= 1;
    n |= n >> 1;
    n |= n >> 2;
    n |= n >> 4;
    n |= n >> 8;
    n |= n >> 16;
    n += 1;
    return n;
}

constexpr inline bool is_power_of_2(int n) {
    return next_power_of_2(n) == n;
}


constexpr inline int ilog2(int n) {
    if (n <= 1) {
        return 0;
    } else{
        return 1 + ilog2(n / 2);
    }
}


constexpr inline int max_pack_size(const int n) {
    if (n <= 0) {
        return 1;
    } else if (n % 4 == 0) {
        return 4;
    } else if (n % 2 == 0) {
        return 2;
    }
    return 1;
}

template <int pack>
__device__ inline void vec_load(const float* ptr, float* dest) {
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
__device__ inline void vec_load_nullable(const float* ptr, float* dest) {
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

template <int pack>
__device__ inline void vec_load_nullable(const half* ptr, half* dest) {
    static_assert(pack == 1 || pack == 2 || pack == 4 || pack == 8, "illegal vector load number");
    // const auto f = [&]() {
    //     if constexpr(pack == 1) {
    //         return *ptr;
    //     } else if constexpr(pack == 2) {
    //         return *((half2*) ptr);
    //     } else if constexpr(pack == 4) {
    //         return *((float2*) ptr);
    //     } else {
    //         return *((float4*) ptr);
    //     }
    // };

    // if (ptr) {
    //     auto v = f();
    //     #pragma unroll
    //     for (int i = 0; i < pack; ++i) {
    //         dest[i] = ((half*) &v)[i];
    //     }
    // } else {
    //     #pragma unroll
    //     for (int i = 0; i < pack; ++i) {
    //         dest[i] = 0.0f;
    //     }
    // }

    if constexpr(pack == 1) {
        if (ptr)
            *dest = *ptr;
        else
            *dest = 0.0f;
    } else if constexpr (pack == 2) {
        // well hopefully this will be optimized away if dest points to registers
        half2 v;
        if (ptr)
            v = *((half2*) ptr);
        else {
            v.x = 0.0f;
            v.y = 0.0f;
        }
        dest[0] = v.x;
        dest[1] = v.y;
    } else if constexpr (pack == 4) {
        if (ptr) {
            float2 v = *((float2*) ptr);
            dest[0] = ((half*) &v)[0];
            dest[1] = ((half*) &v)[1];
            dest[2] = ((half*) &v)[2];
            dest[3] = ((half*) &v)[3];
        } else {
            dest[0] = 0.0f;
            dest[1] = 0.0f;
            dest[2] = 0.0f;
            dest[3] = 0.0f;
        }
    } else {
        float4 v;
        if (ptr) {
            v = *((float4*) ptr);
            #pragma unroll
            for (int i = 0; i < 8; ++i) {
                dest[i] = ((half*) &v)[i];
            }
        }
        else {
            #pragma unroll
            for (int i = 0; i < 8; ++i) {
                dest[i] = 0.0f;
            }
        }
    }
}

// greatest common divisor, aka highest common factor
DevHost constexpr inline int gcd(int x, int y)
{
    if(x < 0)
    {
        return gcd(-x, y);
    }
    else if(y < 0)
    {
        return gcd(x, -y);
    }
    else if(x == y || x == 0)
    {
        return y;
    }
    else if(y == 0)
    {
        return x;
    }
    else if(x > y)
    {
        return gcd(x % y, y);
    }
    else
    {
        return gcd(x, y % x);
    }
}

DevHost constexpr inline int lcm(int x, int y) {
    return (x * y) / gcd(x, y);
}

__device__ inline void block_sync_lds()
{
    asm volatile("\
    s_waitcnt lgkmcnt(0) \n \
    s_barrier \
    " ::);

}

using int32x2_t = __attribute__( (__vector_size__(2 * sizeof(int)) )) int;
using int32x4_t = __attribute__( (__vector_size__(4 * sizeof(int)) )) int;


using float2_t = __attribute__( (__vector_size__(2 * sizeof(float)) )) float;
using float4_t = __attribute__( (__vector_size__(4 * sizeof(float)) )) float;
using float16_t = __attribute__( (__vector_size__(16 * sizeof(float)) )) float;
using half2_t = __attribute__( (__vector_size__(2 * sizeof(_Float16)) )) _Float16;
using half4_t = __attribute__( (__vector_size__(4 * sizeof(_Float16)) )) _Float16;

