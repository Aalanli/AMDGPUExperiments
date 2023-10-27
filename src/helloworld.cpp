#include <assert.h>
#include <bits/floatn-common.h>
#include <hip/amd_detail/amd_hip_runtime.h>
#include <hip/amd_detail/amd_hip_vector_types.h>

#include <stdio.h>
#include <algorithm>
#include <stdlib.h>
#include <iostream>
#include "hip/hip_runtime.h"
// #include "utils.hpp"


#define WIDTH     1024
#define HEIGHT    1024

#define NUM       (WIDTH*HEIGHT)

#define THREADS_PER_BLOCK_X  16
#define THREADS_PER_BLOCK_Y  16
#define THREADS_PER_BLOCK_Z  1

__global__ void 
vectoradd_float(float* __restrict__ a, const float* __restrict__ b, const float* __restrict__ c, int width, int height) { 
    int x = blockDim.x * hipBlockIdx_x + hipThreadIdx_x;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;

    int i = y * width + x;
    if ( i < (width * height)) {
    a[i] = b[i] + c[i];
    }
}

extern "C" bool rocblas_gemm(const float* a, const float* b, float* c, int m, int k, int n);

using namespace std;

template<typename T>
struct DeviceBuf {
    T* ptr;
    int size;

    DeviceBuf(int size) {
        this->size = size;
        (hipMalloc(&ptr, size * sizeof(T)));
    }

    ~DeviceBuf() {
        (hipFree(ptr));
    }

    void zero() {
        (hipMemset(this->ptr, 0, this->size * sizeof(T)));
    }
};

int main() {
    hipDeviceProp_t devProp;
    hipGetDeviceProperties(&devProp, 0);
    cout << " System minor " << devProp.minor << endl;
    cout << " System major " << devProp.major << endl;
    cout << " agent prop name " << devProp.name << endl;

    cout << "hip Device prop succeeded " << endl ;
    {
        DeviceBuf<float> deviceA(NUM);
        DeviceBuf<float> deviceB(NUM);
        DeviceBuf<float> deviceC(NUM);

        hipLaunchKernelGGL(vectoradd_float, 
                        dim3(WIDTH/THREADS_PER_BLOCK_X, HEIGHT/THREADS_PER_BLOCK_Y),
                        dim3(THREADS_PER_BLOCK_X, THREADS_PER_BLOCK_Y),
                        0, 0,
                        deviceA.ptr, deviceB.ptr, deviceC.ptr, WIDTH, HEIGHT);
    }

    {
        int m = 1024;
        int n = 1024;
        int k = 1024;
        DeviceBuf<float> A(m * k);
        DeviceBuf<float> B(k * n);
        DeviceBuf<float> C(m * n);
        
        auto passed = rocblas_gemm(A.ptr, B.ptr, C.ptr, m, k, n);
        cout << "rocblas_gemm " << passed << endl;
    }

}