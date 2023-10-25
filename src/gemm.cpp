/* ************************************************************************
 * Copyright (C) 2019-2022 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell cop-
 * ies of the Software, and to permit persons to whom the Software is furnished
 * to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IM-
 * PLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
 * FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
 * COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
 * IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNE-
 * CTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 *
 * ************************************************************************ */

#include "hip/hip_runtime.h"
#include "rocblas/rocblas.h"
#include <cassert>
#include <math.h>
#include <iostream>
#include <rocblas/internal/rocblas-functions.h>
#include <rocblas/internal/rocblas-types.h>
#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <string>
#include "utils.hpp"

#ifndef CHECK_ROCBLAS_STATUS
#define CHECK_ROCBLAS_STATUS(status)                  \
    if(status != rocblas_status_success)              \
    {                                                 \
        fprintf(stderr, "rocBLAS error: ");           \
        fprintf(stderr,                               \
                "rocBLAS error: '%s'(%d) at %s:%d\n", \
                rocblas_status_to_string(status),     \
                status,                               \
                __FILE__,                             \
                __LINE__);                            \
        exit(EXIT_FAILURE);                           \
    }
#endif

int main(int argc, char** argv)
{
    rocblas_status rstatus = rocblas_status_success;

    typedef float dataType;

    assert(argc == 3 && "M K N");

    const rocblas_int M = std::stoi(argv[1]);
    const rocblas_int K = std::stoi(argv[2]);
    const rocblas_int N = std::stoi(argv[3]);

    const float hAlpha = 1.0f;
    const float hBeta  = 0.0f;

    const rocblas_operation transA = rocblas_operation_none;
    const rocblas_operation transB = rocblas_operation_none;

    rocblas_int lda, ldb, ldc, sizeA, sizeB, sizeC;
    int         strideA1, strideA2, strideB1, strideB2;

    if(transA == rocblas_operation_none)
    {
        lda      = M;
        sizeA    = K * lda;
        strideA1 = 1;
        strideA2 = lda;
    }
    else
    {
        lda      = K;
        sizeA    = M * lda;
        strideA1 = lda;
        strideA2 = 1;
    }
    if(transB == rocblas_operation_none)
    {
        ldb      = K;
        sizeB    = N * ldb;
        strideB1 = 1;
        strideB2 = ldb;
    }
    else
    {
        ldb      = N;
        sizeB    = K * ldb;
        strideB1 = ldb;
        strideB2 = 1;
    }
    ldc   = M;
    sizeC = N * ldc;

    // using rocblas API
    rocblas_handle handle;
    rstatus = rocblas_create_handle(&handle);
    CHECK_ROCBLAS_STATUS(rstatus);


    {
        float* dA;
        float* dB;
        float* dC;

        HIP_ASSERT(hipMalloc((void**)&dA, M * K * sizeof(float)));
        HIP_ASSERT(hipMalloc((void**)&dB, K * N * sizeof(float)));
        HIP_ASSERT(hipMalloc((void**)&dC, M * K * sizeof(float)));

        // enable passing alpha parameter from pointer to host memory
        rstatus = rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host);
        CHECK_ROCBLAS_STATUS(rstatus);

        // asynchronous calculation on device, returns before finished calculations
        rstatus = rocblas_sgemm(
            handle, transA, transB, M, N, K, &hAlpha, dA, lda, dB, ldb, &hBeta, dC, ldc);

        CHECK_ROCBLAS_STATUS(rstatus);
        HIP_ASSERT(hipDeviceSynchronize());
        HIP_ASSERT(hipFree(dA));
        HIP_ASSERT(hipFree(dB));
        HIP_ASSERT(hipFree(dC));
    }
    {
        rocblas_half *dA, *dB, *dC;
        HIP_ASSERT(hipMalloc((void**)&dA, M * K * sizeof(rocblas_half)));
        HIP_ASSERT(hipMalloc((void**)&dB, K * N * sizeof(rocblas_half)));
        HIP_ASSERT(hipMalloc((void**)&dC, M * K * sizeof(rocblas_half)));

        rocblas_half a, b;
        a.data = 0;
        b.data = 0;

        rstatus = rocblas_hgemm(
            handle, transA, transB, M, N, K, &a, (rocblas_half*) dA, lda, (rocblas_half*) dB, ldb, &b, (rocblas_half*) dC, ldc);

        CHECK_ROCBLAS_STATUS(rstatus);
        HIP_ASSERT(hipDeviceSynchronize());
        HIP_ASSERT(hipFree(dA));
        HIP_ASSERT(hipFree(dB));
        HIP_ASSERT(hipFree(dC));
    }

    std::cout << "M, N, K, lda, ldb, ldc = " << M << ", " << N << ", " << K << ", " << lda << ", "
              << ldb << ", " << ldc << std::endl;

    rstatus = rocblas_destroy_handle(handle);
    CHECK_ROCBLAS_STATUS(rstatus);

    return EXIT_SUCCESS;
}