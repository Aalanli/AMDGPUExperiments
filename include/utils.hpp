#pragma once

#include "hip/hip_runtime.h"
#include <stdio.h>
#include <stdlib.h>


#ifndef HIP_ASSERT
#define HIP_ASSERT(error)                    \
    if(error != hipSuccess)                       \
    {                                             \
        fprintf(stderr,                           \
                "hip error: '%s'(%d) at %s:%d\n", \
                hipGetErrorString(error),         \
                error,                            \
                __FILE__,                         \
                __LINE__);                        \
        exit(EXIT_FAILURE);                       \
    }
#endif

