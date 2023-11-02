#include <iostream>

/// multidimensional static shaped tensor
template <typename T, int D1, int... dims>
struct Tensor {
    static constexpr int dim = D1;
    static constexpr int stride = Tensor<T, dims...>::dim * Tensor<T, dims...>::stride;
    T* data;
    Tensor(T* ptr) : data(ptr) {}
    Tensor<T, dims...> operator[](int i) {
        return Tensor<T, dims...>(data + i * stride);
    }
};

template <typename T, int D1>
struct Tensor<T, D1> {
    static constexpr int dim = D1;
    static constexpr int stride = 1;
    T* data;
    Tensor(T* ptr) : data(ptr) {}
    T& operator[](int i) {
        return data[i];
    }
};


int main() {
    float ptr[4 * 3 * 2];
    for (int i = 0; i < 4 * 3 * 2; i++) {
        ptr[i] = 0;
    }
    Tensor<float, 2, 3, 4> t{ptr};

    t[0][0][0] = 1;
    t[0][0][1] = 2;
    t[0][1][2] = 3;

    printf("%f\n", t[0][0][0]);
    printf("%f\n", t[0][0][2]);
    for (int i = 0; i < 4 * 3 * 2; i++) {
        printf("%f ", ptr[i]);
    }
    return 0;
}