#include "mfma_tools.cpp"
#include <cstdlib>
#include <stdio.h>

template <typename L>
void print_layout() {
    int size = L::s1 * L::s2;
    repeat<L::s1, L::s2>([&](int i, int j) {
        printf("%d ", L::index(i, j));
        if (j == L::s2 - 1)
            printf("\n");        
    });
}

template <typename F>
constexpr auto test(F&& f) {
    return [&] (int i, int j) {
        return i * 10 + j + f(i, j);
    };
}

__global__ void test(int* t) {
    using Task = ComposeTask<SpatialTask<2, 1>, ComposeTask<RepeatTask<2, 1>, SpatialTask<1, 32>>>;
    Task::apply(threadIdx.x, [=] __device__ (Index global, Index local) {
        t[global.i * 32 + global.j] = local.i;
    });
}


int main() {
    using L = ReshapeLayout<SwizzleLayout<4, 32>, 8, 16>;
    print_layout<L>();

    auto t = test([](int i, int j) { return i ^ j; });
    printf("%d\n", t(1, 2));
}