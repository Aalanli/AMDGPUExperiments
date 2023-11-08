// tests various ways structs can be used in cuda programs
#include <stdio.h>

template <int N>
struct Smem {
    __device__ float& operator[](int i) {
        __shared__ float smem[N];
        return smem[i];
    }
};

struct Program {
    const float* __restrict__ a;
    float* __restrict__ b;

    const int tid = threadIdx.x;
    // this is not allowed
    // __shared__ float smem[512];
    Smem<512> smem;

    __device__ Program(const float* a, float* b) : a(a), b(b) {}

    __device__ void execute() {
        // __shared__ float smem[512];
        smem[threadIdx.x] = -1;
        smem[threadIdx.x] = a[threadIdx.x];
        b[threadIdx.x] = smem[512 - 1 - threadIdx.x];
    }

    // this is not allowed
    // static __global__ void kernel(const float* a, float* b) {
    //     Program(a, b).execute();
    // }
};

struct Program2 : Program {
    __device__ void execute() {
        // __shared__ float smem[512];
        smem[tid] = a[tid];
        b[tid] = smem[512 - 1 - tid] * 2.0f;
    }
};

template <typename T>
__global__ void kernel(const float* a, float* b) {
    T(a, b).execute();
}

int main() {
    float a[512], b[512];
    for (int i = 0; i < 512; ++i) {
        a[i] = i;
    }

    float* d_a, *d_b;
    cudaMalloc(&d_a, sizeof(a));
    cudaMalloc(&d_b, sizeof(b));

    cudaMemcpy(d_a, a, sizeof(float) * 512, cudaMemcpyHostToDevice);
    kernel<Program><<<1, 512>>>(d_a, d_b);
    cudaMemcpy(b, d_b, sizeof(float) * 512, cudaMemcpyDeviceToHost);
    // print b
    for (int i = 0; i < 512; ++i) {
        printf("%f ", b[i]);
    }
    printf("\n");
    

    cudaFree(d_a);
    cudaFree(d_b);
}
