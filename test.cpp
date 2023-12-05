#define DevHost 
#include <stdlib.h>
#include <stdio.h>
#include <type_traits>


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

#define _BLOCK_K 32
#define _BLOCK_N 64

struct Test {
    Test() {
        printf("Test\n");
    }
};

int main() {
    // constexpr int inner_pack = 4;
    // using InnerPackedLayout = ComposeLayout<RowLayout<1, inner_pack / 2>, TransposeLayout<RowLayout<1, 2>>>;
    // using SharedMemLayoutB = ComposeLayout<SwizzleLayout<_BLOCK_K / 2, _BLOCK_N / (inner_pack / 2)>, InnerPackedLayout>;
    // repeat<_BLOCK_K, _BLOCK_N>([](int i, int j) {
    //     printf("%d ", SharedMemLayoutB::index(i, j));
    //     if (j == _BLOCK_N - 1) {
    //         printf("\n");
    //     }
    // }); 
    Test t;
}