#include <hip/hip_runtime.h>
#include <stdio.h>
// #include <Python.h>

int main() {
    hipDeviceProp_t dev;
    auto _ = hipGetDeviceProperties(&dev, 0);
    printf("smem %zu \n", dev.sharedMemPerBlock);
    printf("regs %d \n", dev.regsPerBlock);
    printf("coorperative launch %d \n", dev.cooperativeLaunch);
    printf("concurrent kernels %d \n", dev.concurrentKernels);
}