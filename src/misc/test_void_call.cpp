#include "utils.hpp"
#include <hip/amd_detail/amd_hip_runtime.h>
#include <vector>

__global__ void test(int a, float* __restrict__ b) {}


template <typename... Args>
class KernelLauncher {
    using FnPtr = void (*)(Args...);
    std::vector<FnPtr> kernels;
public:
    KernelLauncher(std::vector<FnPtr> kernels) : kernels(kernels) {}

    void operator()(dim3 grid, dim3 block, Args... args) {
        for (auto kernel : kernels) {
            hipLaunchKernelGGL(kernel, grid, block, 0, 0, args...);
        }
    }
};


int main() {
    std::vector<void(*)(int, float*)> kernels = {
        test
    };
    KernelLauncher<int, float*> launcher(kernels);
    launcher(dim3(1), dim3(1), 1, nullptr);
}