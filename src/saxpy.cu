#ifndef LAUNCH_NAME
#define LAUNCH_NAME saxpy
#endif

#ifndef BLOCKSIZE
#define BLOCKSIZE 256
#endif

#ifndef REPEATS
#define REPEATS 4
#endif

__global__ void saxpy_kernel(
    const float* __restrict__ a,
    const float* __restrict__ b,
    float* __restrict__ c,
    int n, int d) 
{
    int bid = blockIdx.x;
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;

    while (tid < d) {
        c[bid * d + tid] = a[bid * d + tid] + b[bid * d + tid];
        tid += stride;
    }
}

extern "C" bool LAUNCH_NAME(float* a, float* b, float* c, int n, int d);

bool LAUNCH_NAME(float* a, float* b, float* c, int n, int d) {
    saxpy_kernel<<<dim3(d / BLOCKSIZE / REPEATS, n), BLOCKSIZE>>>(a, b, c, n, d);
    return true;
}

