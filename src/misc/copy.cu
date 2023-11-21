#include <cstdio>
#include <string>
#include <assert.h>
#include <chrono>


void __global__ copy(const float* __restrict__ a, float* __restrict__ b, const int n) {
    const int stride = blockDim.x * gridDim.x;
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    while (i < n) {
        b[i] = a[i];
        i += stride;
    }
}

void __global__ copyf4(const float4* __restrict__ a, float4* __restrict__ b, const int n) {
    const int stride = blockDim.x * gridDim.x;
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    while (i < n) {
        b[i] = a[i];
        i += stride;
    }
}


int main(int argc, char** argv) {
    const int repeats = 1;
    printf("%d\n", argc);
    // assert(argc == 1);
    const int n = std::stoi(argv[1]);
    printf("copying %i floats\n", n);

    float* a;
    float* b;
    cudaMalloc(&a, n * sizeof(float));
    cudaMalloc(&b, n * sizeof(float));

    int threads = 1024;
    int blocks = (n + threads - 1) / (threads * repeats);

    {
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaDeviceSynchronize();

        cudaEventRecord(start);
        copy<<<blocks, threads>>>(a, b, n);
        cudaEventRecord(stop);
        cudaDeviceSynchronize();
        float t;
        cudaEventElapsedTime(&t, start, stop);

        cudaError_t error = cudaGetLastError();
        if (error != cudaSuccess) {
            printf("CUDA error: %s\n", cudaGetErrorString(error));
            exit(-1);
        }

        printf("time: %f ms\n", t);
        printf("bandwidth: %f GB/s\n", 1e-9 * n * sizeof(float) / (t / 1000));
    }

    if (n % 4 == 0) {
        float4* a4 = (float4*)a;
        float4* b4 = (float4*)b;
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaDeviceSynchronize();
        cudaEventRecord(start);

        blocks = (n / 4 + threads - 1) / (threads * repeats);
        copyf4<<<blocks, threads>>>(a4, b4, n / 4);
        cudaEventRecord(stop);
        cudaDeviceSynchronize();
        float t;
        cudaEventElapsedTime(&t, start, stop);
        printf("time: %f ms\n", t);
        printf("bandwidth: %f GB/s\n", 1e-9 * n * sizeof(float) / (t / 1000));

        cudaError_t error = cudaGetLastError();
        if (error != cudaSuccess) {
            printf("CUDA error: %s\n", cudaGetErrorString(error));
            exit(-1);
        }
    }

    cudaFree(a);
    cudaFree(b);
}
