// #include <hip/amd_detail/amd_warp_functions.h>
// #include <stdio.h>
#include "hip/hip_runtime.h"
// #include <hip/amd_detail/amd_hip_runtime.h>
// #include <hip/amd_detail/amd_hip_vector_types.h>
// #include <hip/hip_runtime_api.h>



__global__ void test_warp_reduce_shfl_down(int* a, int* b) {
    int tid = threadIdx.x;
    int lane = tid % warpSize;
    
    int data = a[tid + blockIdx.x * blockDim.x];
    for (int id = 0; id < warpSize; id++) {
            if (lane == id) {
                printf("%d ", data);
            }
            __syncthreads();
        }
        if (lane == 0)
            printf("\n");
    for (int i = warpSize / 2; i > 0; i = i / 2) {
        data += __shfl_down(data, i);
        for (int id = 0; id < warpSize; id++) {
            if (lane == id) {
                printf("%d ", data);
            }
            __syncthreads();
        }
        if (lane == 0)
            printf("\n");
    }
    b[tid + blockIdx.x * blockDim.x] = data;
}

__global__ void test_warp_reduce_shfl_xor(int* a, int* b) {
    int tid = threadIdx.x;
    int lane = tid % warpSize;
    
    int data = a[tid + blockIdx.x * blockDim.x];
    for (int i = 1; i < warpSize; i *= 2) {
        data += __shfl_xor(data, i);
        for (int id = 0; id < warpSize; id++) {
            if (lane == id) {
                printf("%d ", data);
            }
            __syncthreads();
        }
        if (lane == 0)
            printf("\n");
    }
    b[tid + blockIdx.x * blockDim.x] = data;
}


int test() {
    int in[64];
    int warpsz = 32;
    for (int i = 0; i < 64; i++) {
        in[i] = i;
    }
    int *d_a, *d_b;
    hipMalloc(&d_a, sizeof(int) * warpsz);
    hipMalloc(&d_b, sizeof(int) * warpsz);
    hipMemcpy(d_a, in, sizeof(int) * warpsz, hipMemcpyHostToDevice);

    int res[warpsz];
    hipLaunchKernelGGL(test_warp_reduce_shfl_down, dim3(1), dim3(warpsz), 0, 0, d_a, d_b);
    hipDeviceSynchronize();
    hipMemcpy(res, d_b, sizeof(int) * warpsz, hipMemcpyDeviceToHost);
    for (int i = 0; i < warpsz; i++) {
        printf("%d ", res[i]);
    }
    printf("\n");
    hipLaunchKernelGGL(test_warp_reduce_shfl_xor, dim3(1), dim3(warpsz), 0, 0, d_a, d_b);
    hipDeviceSynchronize();
    hipMemcpy(res, d_b, sizeof(int) * warpsz, hipMemcpyDeviceToHost);
    for (int i = 0; i < warpsz; i++) {
        printf("%d ", res[i]);
    }
    printf("\n");
    hipFree(d_a);
    hipFree(d_b);
}