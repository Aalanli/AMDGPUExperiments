#include "hip_utils.hpp"
#include "amd_buffer_addressing.hpp"


template <typename T, int N>
__global__ void copy(const T* __restrict__ src, T* dest, int src_len, int dest_len) {
    T regs[N];

    int warp = threadIdx.x / 64;
    const T* warp_ptr = src + warp * 64 * N + blockIdx.x * blockDim.x * N;

    int lane_offset = (threadIdx.x % 64) * N;
    amd_buffer_load_invalid_element_set_zero<int, N>(
        warp_ptr, lane_offset, lane_offset < src_len, src_len, &regs[0]);

    
    int tid = N * (threadIdx.x + blockIdx.x * blockDim.x);
    if (tid < dest_len) {
        for (int i = 0; i < N; ++i) {
            dest[tid + i] = regs[i];
        }
    }
}


template <typename T, int N, int M, int K>
__global__ void copy_strided(const T* __restrict__ src, T* dest, int m1, int k1) {
    T regs[N];

    int lane = threadIdx.x % 64;
    constexpr int threads_per_row = K / N;
    int offset_k = (lane % threads_per_row) * N;
    int offset_m = lane / threads_per_row;

    amd_buffer_load_invalid_element_set_zero<int, N>(
        src, offset_k + offset_m * k1, offset_k < k1 && offset_m < m1, m1 * k1, &regs[0]);
    
    for (int i = 0; i < N; ++i) {
        dest[offset_k + offset_m * k1 + i] = regs[i];
    }

}


int main() {
    {
        constexpr int N = 4;
        constexpr int src_len = 64;
        constexpr int dest_len = src_len * N;

        int src[src_len];
        int dest[dest_len];

        for (int i = 0; i < src_len; ++i) {
            src[i] = i;
        }
        int* d_src;
        int* d_dest;

        hipMalloc(&d_src, sizeof(src));
        hipMalloc(&d_dest, sizeof(dest));
        hipMemset(d_dest, -1, sizeof(dest));

        // copy to device
        hipMemcpy(d_src, src, sizeof(src), hipMemcpyHostToDevice);
        hipMemcpy(d_dest, dest, sizeof(dest), hipMemcpyHostToDevice);

        auto kernel = copy<int, N>;
        hipLaunchKernelGGL(kernel, dim3(1), dim3(64), 0, 0, d_src, d_dest, src_len, dest_len);

        // copy back to host
        hipMemcpy(dest, d_dest, sizeof(dest), hipMemcpyDeviceToHost);

        for (int i = 0; i < dest_len; ++i) {
            printf("%d ", dest[i]);
        }
        printf("\n");

        hipFree(src);
        hipFree(dest);
    }

    {
        constexpr int N = 4;
        constexpr int M = 8;
        constexpr int K = 16;

        int src[32 * 32];
        int dest[32 * 32];

        for (int i = 0; i < 1024; ++i) {
            src[i] = i;
        }

        int* d_src;
        int* d_dest;

        hipMalloc(&d_src, sizeof(src));
        hipMalloc(&d_dest, sizeof(dest));
        hipMemset(d_dest, -1, sizeof(dest));

        // copy to device
        hipMemcpy(d_src, src, sizeof(src), hipMemcpyHostToDevice);

        auto kernel = copy_strided<int, N, M, K>;
        hipLaunchKernelGGL(kernel, dim3(1), dim3(64), 0, 0, d_src, d_dest, 32, 32);

        // copy back to host
        hipMemcpy(dest, d_dest, sizeof(dest), hipMemcpyDeviceToHost);

        for (int i = 0; i < 32; ++i) {
            for (int j = 0; j < 32; ++j) {
                printf("%d ", dest[i * 32 + j]);
            }
            printf("\n");
        }

        hipFree(src);
        hipFree(dest);

    }
}
