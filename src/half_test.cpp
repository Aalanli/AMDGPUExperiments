#include "hip_utils.hpp"
// #include <hip/amd_detail/hip_fp16_gcc.h>
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <iostream>

__global__ void test(half* a, half* b) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    a[tid] = b[tid] * half(2.0f);
}

int main() {
    int n = 1024;
    half* a = (half*)malloc(n * sizeof(half));
    half* b = (half*)malloc(n * sizeof(half));
    half* c = (half*)malloc(n * sizeof(half));
    for (int i = 0; i < n; i++) {
        a[i] = 1.0f;
        b[i] = 2.0f;
    }
    half* d_a;
    half* d_b;
    hipMalloc(&d_a, n * sizeof(half));
    hipMalloc(&d_b, n * sizeof(half));
    hipMemcpy(d_a, a, n * sizeof(half), hipMemcpyHostToDevice);
    hipMemcpy(d_b, b, n * sizeof(half), hipMemcpyHostToDevice);
    hipLaunchKernelGGL(test, dim3(n / 256), dim3(256), 0, 0, d_a, d_b);
    hipMemcpy(c, d_a, n * sizeof(half), hipMemcpyDeviceToHost);
    for (int i = 0; i < n; i++) {
        std::cout << (float) c[i] << " ";
    }
    return 0;
}