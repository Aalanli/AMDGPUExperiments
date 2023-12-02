#pragma once

#include "hip_utils.hpp"


template <int S1, int S2>
struct RowLayout {
    static constexpr int s1 = S1;
    static constexpr int s2 = S2;

    static DevHost inline int index(int i, int j) {
        return i * S2 + j;
    }
};

template <int S1, int S2>
struct SwizzleLayout {
    static constexpr int s1 = S1;
    static constexpr int s2 = S2;

    static_assert(is_power_of_2(S2), "");
    static DevHost inline int index(int i, int j) {
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

    static DevHost inline int index(int i, int j) {
        return L::index(j, i);
    }
};

template <typename L1, typename L2>
struct ComposeLayout {
    static constexpr int s1 = L1::s1 * L2::s1;
    static constexpr int s2 = L1::s2 * L2::s2;
    
    static DevHost inline int index(int i, int j) {
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

    DevHost inline LayoutAccessor(T* ptr) : ptr(ptr) {}

    DevHost inline T* index(int i, int j) {
        return ptr + L::index(i, j);
    }
};

template <typename T>
struct RowAccessor {
    using AccT = T;
    T* ptr;
    int a, b;

    // RowAccessor(T* ptr, int a, int b) : ptr(ptr), a(a), b(b) {}

    DevHost inline T* index(int i, int j) {
        if (i < a && j < b) {
            return ptr + offset(i, j);
        }
        return nullptr;
    }

    DevHost inline int offset(int i, int j) {
        return i * b + j;
    }
};

template <typename L>
struct OffsetAccessor {
    using AccT = typename L::AccT;
    L inner;
    int offset_a, offset_b;

    // OffsetAccessor(L inner, int offset_a, int offset_b) : inner(inner), offset_a(offset_a), offset_b(offset_b) {} 

    DevHost inline AccT* index(int i, int j) {
        return inner.index(i + offset_a, j + offset_b);
    }

    DevHost inline int offset(int i, int j) {
        return inner.offset(i + offset_a, j + offset_b);
    }

    DevHost inline void set_offset(int a, int b) {
        offset_a = a;
        offset_b = b;
    }


    DevHost inline void inc_offset(int a, int b) {
        offset_a += a;
        offset_b += b;
    }
};

template <typename A>
__device__ void print_acc(A &acc) {
    __syncthreads();
    if (threadIdx.x == 0) {
        repeat<A::s1, A::s2>([&](int i, int j) {
            printf("%f ", (float) *acc.index(i, j));
            if (j == A::s2 - 1) {
                printf("\n");
            }

        });
    }
    __syncthreads();
}
