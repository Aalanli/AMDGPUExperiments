#include "hip_utils.hpp"
#include <cassert>
#include <cstdio>
#include <hip/amd_detail/amd_hip_runtime.h>

template <typename T,
          int  DIM_M,
          int  DIM_N,
          int  BLK_M,
          int  BLK_N,
          int  BLK_K,
          int  DIM_M_A,
          int  DIM_M_B,
          int  alpha,
          int  beta,
          char TRANS_A,
          char TRANS_B>
KERNEL(DIM_M * DIM_N) rocblas_gemm_kernel(
    int M,
    int N,
    int K,
    const T* __restrict__ dA,
    int lda,
    const T* __restrict__ dB,
    int ldb,
    T* __restrict__ dC,
    int ldc)
{
    const int DIM_N_A = DIM_M * DIM_N / DIM_M_A;
    const int DIM_N_B = DIM_M * DIM_N / DIM_M_B;
    int thx  = threadIdx.x; // thread's m position in C
    int thy  = threadIdx.y; // thread's n position in C
    int idt  = DIM_M * thy + thx; // thread's number
    int blx  = blockIdx.x; // block's m position
    int bly  = blockIdx.y; // block's n position
    // int blz  = blockIdx.z; // block's matrix in the batch
    int thxA = idt % DIM_M_A; // thread's m position for loading A
    int thyA = idt / DIM_M_A; // thread's n position for loading A
    int thxB = idt % DIM_M_B; // thread's m position for loading B
    int thyB = idt / DIM_M_B; // thread's n position for loading B

    __shared__ T sA[BLK_K][BLK_M]; // shared memory for A
    __shared__ T sB[BLK_N][BLK_K]; // shared memory for B
    T            rC[BLK_N / DIM_N][BLK_M / DIM_M]; // registers for C

    size_t coord_A, coord_B;
    if(TRANS_A == 'N')
        coord_A = (blx * BLK_M + thxA) + thyA * size_t(lda);
    else if(TRANS_A == 'T' || TRANS_A == 'C')
        coord_A = (blx * BLK_M + thxA) * size_t(lda) + thyA;
    if(TRANS_B == 'N')
        coord_B = (bly * BLK_N + thyB) * size_t(ldb) + thxB;
    else if(TRANS_B == 'T' || TRANS_B == 'C')
        coord_B = (bly * BLK_N + thyB) + thxB * size_t(ldb);

    for(int n = 0; n < BLK_N / DIM_N; ++n)
        for(int m = 0; m < BLK_M / DIM_M; ++m)
            rC[n][m] = 0.0;

    int kk = 0;
    for(; kk < K; kk += BLK_K)
    {
        for(int n = 0; n < BLK_K; n += DIM_N_A)
            for(int m = 0; m < BLK_M; m += DIM_M_A)
                if(TRANS_A == 'N')
                    sA[n + thyA][m + thxA] = dA[coord_A + (n * size_t(lda) + m)];
                else if(TRANS_A == 'T')
                    sA[n + thyA][m + thxA] = dA[coord_A + (n + m * size_t(lda))];
                // else if(TRANS_A == 'C')
                //     sA[n + thyA][m + thxA] = conj(dA[coord_A + (n + m * size_t(lda))]);

        for(int n = 0; n < BLK_N; n += DIM_N_B)
            for(int m = 0; m < BLK_K; m += DIM_M_B)
                if(TRANS_B == 'N')
                    sB[n + thyB][m + thxB] = dB[coord_B + (n * size_t(ldb) + m)];
                else if(TRANS_B == 'T')
                    sB[n + thyB][m + thxB] = dB[coord_B + (n + m * size_t(ldb))];
                // else if(TRANS_B == 'C')
                //     sB[n + thyB][m + thxB] = conj(dB[coord_B + (n + m * size_t(ldb))]);

        __syncthreads();

        for(int k = 0; k < BLK_K; ++k)
            for(int n = 0; n < BLK_N / DIM_N; ++n)
                for(int m = 0; m < BLK_M / DIM_M; ++m)
                    rC[n][m] += sA[k][m * DIM_M + thx] * sB[n * DIM_N + thy][k];

        __syncthreads();

        if(TRANS_A == 'N')
            coord_A += BLK_K * size_t(lda);
        else if(TRANS_A == 'T' || TRANS_A == 'C')
            coord_A += BLK_K;

        if(TRANS_B == 'N')
            coord_B += BLK_K;
        else if(TRANS_B == 'T' || TRANS_B == 'C')
            coord_B += BLK_K * size_t(ldb);
    }

    for(int n = 0; n < BLK_N / DIM_N; ++n)
    {
        for(int m = 0; m < BLK_M / DIM_M; ++m)
        {
            int coord_dCm = blx * BLK_M + m * DIM_M + thx;
            int coord_dCn = bly * BLK_N + n * DIM_N + thy;

            if(alpha == 1 && beta == 1)
            {
                dC[coord_dCn * size_t(ldc) + coord_dCm] += rC[n][m];
            }
            else if(alpha == 1 && beta == -1)
            {
                dC[coord_dCn * size_t(ldc) + coord_dCm]
                    = -dC[coord_dCn * size_t(ldc) + coord_dCm] + rC[n][m];
            }
            else if(alpha == -1 && beta == 0)
            {
                dC[coord_dCn * size_t(ldc) + coord_dCm] = -rC[n][m];
            }
            else if(alpha == 1 && beta == 0)
            {
                dC[coord_dCn * size_t(ldc) + coord_dCm] = rC[n][m];
            }
        }
    }
}


template <typename T,
          int  DIM_M,
          int  DIM_N,
          int  BLK_M,
          int  BLK_N,
          int  BLK_K,
          int  DIM_K_A,
          int  DIM_N_B>
KERNEL(DIM_M * DIM_N) rocblas_gemm_kernelv2(
    int M,
    int N,
    int K,
    const T* __restrict__ dA, // M x K
    const T* __restrict__ dB, // K x N
    T* __restrict__ dC)
{
    const int DIM_M_A = DIM_M * DIM_N / DIM_K_A;
    const int DIM_K_B = DIM_M * DIM_N / DIM_N_B;
    int thx  = threadIdx.x; // thread's m position in C
    int thy  = threadIdx.y; // thread's n position in C
    int idt  = DIM_M * thy + thx; // thread's number
    int blx  = blockIdx.x; // block's m position
    int bly  = blockIdx.y; // block's n position
    // int blz  = blockIdx.z; // block's matrix in the batch
    int thxA = idt % DIM_K_A; // thread's row position for loading A
    int thyA = idt / DIM_K_A; // thread's col position for loading A
    int thxB = idt % DIM_N_B; // thread's row position for loading B
    int thyB = idt / DIM_N_B; // thread's col position for loading B

    __shared__ T sA[BLK_M][BLK_K]; // shared memory for A
    __shared__ T sB[BLK_K][BLK_N]; // shared memory for B
    T            rC[BLK_M / DIM_M][BLK_N / DIM_N]; // registers for C

    int coord_Am = blx * BLK_M + thyA;
    int coord_Ak = thxA;
    int coord_Bk = thyB;
    int coord_Bn = bly * BLK_N + thxB;

    for(int n = 0; n < BLK_N / DIM_N; ++n)
        for(int m = 0; m < BLK_M / DIM_M; ++m)
            rC[m][n] = 0.0;

    for(int kk = 0; kk < K; kk += BLK_K) {
        for(int k = 0; k < BLK_K; k += DIM_K_A) {
            for(int m = 0; m < BLK_M; m += DIM_M_A) {
                if (coord_Ak + k < K && coord_Am + m < M) {
                    sA[m + thyA][k + thxA] = dA[coord_Ak + k + (coord_Am + m) * K];
                } else {
                    sA[m + thyA][k + thxA] = 0.0;
                }
            }
        }
                
        for(int n = 0; n < BLK_N; n += DIM_N_B)
            for(int k = 0; k < BLK_K; k += DIM_K_B)
                if (coord_Bk + k < K && coord_Bn + n < N)
                    sB[k + thyB][n + thxB] = dB[(coord_Bk + k) * N + coord_Bn + n];
                else
                    sB[k + thyB][n + thxB] = 0.0;

        __syncthreads();

        for(int k = 0; k < BLK_K; ++k) {
            for(int n = 0; n < BLK_N / DIM_N; ++n) {
                for(int m = 0; m < BLK_M / DIM_M; ++m) {
                    T t = sA[m * DIM_M + thx][k] * sB[k][n * DIM_N + thy];
                    // assert(t == 1);
                    rC[m][n] += t;
                }
            }
        }
        __syncthreads();

        coord_Ak += BLK_K;
        coord_Bk += BLK_K;
    }

    for(int n = 0; n < BLK_N / DIM_N; ++n) {
        for(int m = 0; m < BLK_M / DIM_M; ++m) {
            int coord_dCm = blx * BLK_M + m * DIM_M + thx;
            int coord_dCn = bly * BLK_N + n * DIM_N + thy;

            if (coord_dCm < M && coord_dCn < N)
                dC[coord_dCm * N + coord_dCn] = rC[m][n];
        }
    }
}

template <typename T,
          int  DIM_M,
          int  DIM_N,
          int  BLK_M,
          int  BLK_N,
          int  BLK_K,
          int  DIM_K_A,
          int  DIM_N_B>
KERNEL(DIM_M * DIM_N) rocblas_gemm_kernelv3(
    int M,
    int N,
    int K,
    const T* __restrict__ dA, // M x K
    const T* __restrict__ dB, // K x N
    T* __restrict__ dC)
{
    const int DIM_M_A = DIM_M * DIM_N / DIM_K_A;
    const int DIM_K_B = DIM_M * DIM_N / DIM_N_B;
    int thx  = threadIdx.x; // thread's n position in C
    int thy  = threadIdx.y; // thread's m position in C
    int idt  = DIM_N * thy + thx; // thread's number
    int blx  = blockIdx.x; // block's m position
    int bly  = blockIdx.y; // block's n position
    // int blz  = blockIdx.z; // block's matrix in the batch
    int thxA = idt % DIM_K_A; // thread's row position for loading A
    int thyA = idt / DIM_K_A; // thread's col position for loading A
    int thxB = idt % DIM_N_B; // thread's row position for loading B
    int thyB = idt / DIM_N_B; // thread's col position for loading B

    __shared__ T sA[BLK_M][BLK_K]; // shared memory for A
    __shared__ T sB[BLK_K][BLK_N]; // shared memory for B
    T            rC[BLK_M / DIM_M][BLK_N / DIM_N]; // registers for C

    int coord_Am = blx * BLK_M + thyA;
    int coord_Ak = thxA;
    int coord_Bk = thyB;
    int coord_Bn = bly * BLK_N + thxB;

    for(int n = 0; n < BLK_N / DIM_N; ++n)
        for(int m = 0; m < BLK_M / DIM_M; ++m)
            rC[m][n] = 0.0;

    for(int kk = 0; kk < K; kk += BLK_K) {
        for(int k = 0; k < BLK_K; k += DIM_K_A) {
            for(int m = 0; m < BLK_M; m += DIM_M_A) {
                if (coord_Ak + k < K && coord_Am + m < M) {
                    sA[m + thyA][k + thxA] = dA[coord_Ak + k + (coord_Am + m) * K];
                } else {
                    sA[m + thyA][k + thxA] = 0.0;
                }
            }
        }
                
        for(int n = 0; n < BLK_N; n += DIM_N_B)
            for(int k = 0; k < BLK_K; k += DIM_K_B)
                if (coord_Bk + k < K && coord_Bn + n < N)
                    sB[k + thyB][n + thxB] = dB[(coord_Bk + k) * N + coord_Bn + n];
                else
                    sB[k + thyB][n + thxB] = 0.0;

        __syncthreads();

        for(int k = 0; k < BLK_K; ++k) {
            for(int n = 0; n < BLK_N / DIM_N; ++n) {
                for(int m = 0; m < BLK_M / DIM_M; ++m) {
                    T t = sA[m * DIM_M + thy][k] * sB[k][n * DIM_N + thx];
                    // assert(t == 1);
                    rC[m][n] += t;
                }
            }
        }
        __syncthreads();

        coord_Ak += BLK_K;
        coord_Bk += BLK_K;
    }

    for(int n = 0; n < BLK_N / DIM_N; ++n) {
        for(int m = 0; m < BLK_M / DIM_M; ++m) {
            int coord_dCm = blx * BLK_M + m * DIM_M + thy;
            int coord_dCn = bly * BLK_N + n * DIM_N + thx;

            if (coord_dCm < M && coord_dCn < N)
                dC[coord_dCm * N + coord_dCn] = rC[m][n];
        }
    }
}

#ifndef LAUNCH_NAME
#define LAUNCH_NAME rocblas_gemm
#endif

#ifndef BLOCKSIZE_M
#define BLOCKSIZE_M 64
#endif

#ifndef BLOCKSIZE_N
#define BLOCKSIZE_N 64
#endif

#ifndef BLOCKSIZE_K
#define BLOCKSIZE_K 16
#endif

#ifndef WARPSZ_M
#define WARPSZ_M 32
#endif

#ifndef WARPSZ_N
#define WARPSZ_N 8
#endif

#ifndef READ_A_DIM
#define READ_A_DIM 64
#endif

#ifndef READ_B_DIM
#define READ_B_DIM 64
#endif

#ifndef TYPE
#define TYPE float
#endif

bool check_smem_use(int use) {
    int smem_size;
    hipDeviceGetAttribute(&smem_size, hipDeviceAttributeMaxSharedMemoryPerBlock, 0);
    return smem_size > use;
}

EXPORT bool LAUNCH_NAME(const TYPE* a, const TYPE* b, TYPE* c, int m, int k, int n, int version) {
    auto kernelv1 = rocblas_gemm_kernelv2<TYPE, WARPSZ_M, WARPSZ_N, BLOCKSIZE_M, BLOCKSIZE_N, BLOCKSIZE_K, READ_A_DIM, READ_B_DIM>;
    auto kernelv2 = rocblas_gemm_kernelv3<TYPE, WARPSZ_M, WARPSZ_N, BLOCKSIZE_M, BLOCKSIZE_N, BLOCKSIZE_K, READ_A_DIM, READ_B_DIM>;
    
    int used_smem = BLOCKSIZE_K * (BLOCKSIZE_M + BLOCKSIZE_N) * sizeof(TYPE);
    if (!check_smem_use(used_smem)) {
        printf("Not enough shared memory: %d\n", used_smem);
        return false;
    }
    if (version != 1 && version != 2)
        return false;
    if (READ_A_DIM > BLOCKSIZE_K)
        return false;
    if (READ_B_DIM > BLOCKSIZE_N)
        return false;
    int blocks_m = cdiv(m, BLOCKSIZE_M);
    int blocks_n = cdiv(n, BLOCKSIZE_N);
    if (version == 1) 
        hipLaunchKernelGGL(kernelv1, dim3(blocks_m, blocks_n), dim3(WARPSZ_M, WARPSZ_N), 0, 0, m, n, k, a, b, c);
    else if (version == 2)
        hipLaunchKernelGGL(kernelv2, dim3(blocks_m, blocks_n), dim3(WARPSZ_N, WARPSZ_M), 0, 0, m, n, k, a, b, c);
    auto error = hipGetLastError();
    if (error != hipSuccess) {
        printf("Error: %s\n", hipGetErrorString(error));
        return false;
    }
    return true;
}

