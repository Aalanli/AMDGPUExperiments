/// constexpr max
template <int A, int B>
struct Max {
    static constexpr int value = A > B ? A : B;
};

__host__ __device__ constexpr inline unsigned int next_power_of_2(const unsigned int a) {
    unsigned int v = a;
    v--;
    v |= v >> 1;
    v |= v >> 2;
    v |= v >> 4;
    v |= v >> 8;
    v |= v >> 16;
    v += 1;
    return v;
}

inline int hmax(int a, int b) {
    return a > b ? a : b;
}

__host__ __device__ inline int cdiv(int a, int b) {
    return (a + b - 1) / b;
}


#ifndef EXPORT
#define EXPORT extern "C" __attribute__((visibility("default")))
#endif

#define KERNEL(lb_) static __global__ __launch_bounds__((lb_)) void


__device__ void debug_print_block(float v, dim3 strides) {
    int tid = threadIdx.x;// + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y;
    int x_tile = tid % strides.x;
    int y_tile = (tid / strides.x) % strides.y;
    int z_tile = tid / (strides.x * strides.y);

    int n_threads = blockDim.x * blockDim.y * blockDim.z;
    for (int i = 0; i < n_threads; ++i) {
        int ix = i % strides.x;
        int iy = (i / strides.x) % strides.y;
        int iz = i / (strides.x * strides.y);

        if (ix == x_tile && iy == y_tile && iz == z_tile) {
            // printf("[%d %d]: ", ix, iy);
            if (ix == 0 && iy == 0) {
                printf("z=%d [%d, %d]\n", iz, strides.y, strides.x);
            }
            printf("%f ", v);
            if (ix == strides.x - 1) {
                printf("\n");
            }
            if (ix == strides.x - 1 && iy == strides.y - 1) {
                printf("----------------\n");
            }
        }
        __syncthreads();
    }
}


/// multidimensional static shaped tensor
template <typename T, int D1, int... dims>
struct Tensor {
    static constexpr int dim = D1;
    static constexpr int stride = Tensor<T, dims...>::dim * Tensor<T, dims...>::stride;
    T* data;
    __host__ __device__ inline Tensor(T* ptr) : data(ptr) {}
    __host__ __device__ inline Tensor<T, dims...> operator[](int i) {
        return Tensor<T, dims...>(data + i * stride);
    }
};

template <typename T, int D1>
struct Tensor<T, D1> {
    static constexpr int dim = D1;
    static constexpr int stride = 1;
    T* data;
    __host__ __device__ inline Tensor(T* ptr) : data(ptr) {}
    __host__ __device__ inline T& operator[](int i) {
        return data[i];
    }
};