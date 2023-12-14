
template <int I1, int I2>
struct Sh {
    enum { i1 = I1 };
    enum { i2 = I2 };
    static const int i3 = I1 + I2;
};

struct A {
    const int t2 = Sh<32, 32>::i3;
};

template <typename T, int T1, int T2>
struct Tensor {
    T a[T1][T2];
};

template <typename T, int T1, int T2, typename F>
__device__ Tensor<T, T1, T2> assign(F&& f) {

    Tensor<T, T1, T2> t;
    for (int i = 0; i < 32; ++i) {
        for (int j = 0; j < 32; ++j)
            t.a[i][j] = f(i, j);
    }
}

__global__ void test() {
    int gsize = blockDim.x;
    auto t = assign<float, 32, 32>([&](int i, int j) { return i * gsize + j; });
    auto t1 = [](int i, int j) { return i + j; };
    int (*f)(int, int) = t1;
    auto t2 = assign<float, 32, 32>(f);
}

int main() {
}