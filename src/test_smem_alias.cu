#include <cstdio>
#include <stdio.h>


__device__ __forceinline__ void test(char* a) {
    __shared__ char s[0xc000];
    int tid = threadIdx.x;
    s[tid] = a[tid];
    a[tid] = s[(tid + 1) % blockDim.x];
}

__global__ void alias_test(char *a, char *b) {
    __shared__ char s[0xc000];
    int tid = threadIdx.x;
    s[tid] = a[tid];
    a[tid] = s[(tid + 1) % blockDim.x];
    __syncthreads();
    test(b);
}


int main() {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("Shared memory per block: %zu\n", prop.sharedMemPerBlock);

    char *a, *b;
    cudaMalloc(&a, 49152 * sizeof(char));
    cudaMalloc(&b, 49152 * sizeof(char));

    alias_test<<<1, 1024>>>(a, b);
    cudaDeviceSynchronize();
    cudaFree(a);
    cudaFree(b);
    // printf("%d", sizeof(short));

}