#include "utils.hpp"

template <int F_OP>
__device__ float arith_workload(float v) {
    for (int i = 0; i < F_OP; ++i) {
        v = v * v + v;
    }
}

template <int F_OP, int NPipeline>
__global__ void reg_gmem_pipeline(float *in, float *out, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) {
        float v = in[idx];
        arith_workload<F_OP>(v);
        out[idx] = v;
    }
}