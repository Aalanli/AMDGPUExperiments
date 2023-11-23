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

int main() {
    using L = ComposeLayout<SwizzleLayout<4, 4>, RowLayout<1, 4>>;
    print_layout<L>();

    using L1 = TransposeLayout<L>;
    print_layout<L1>();
}