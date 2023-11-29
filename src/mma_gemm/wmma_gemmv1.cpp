#include "hip_utils.hpp"
#include "mfma_tools.cpp"

#include "launch_defs.hpp"


EXPORT bool LAUNCH_NAME(
    const float* A, const float* B, float* C,
    int M, int K, int N, int ver
) {
    return true;
}