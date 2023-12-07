#include <hip/hip_runtime.h>
#include <stdio.h>

int main() {
    hipDeviceProp_t prop;
    hipGetDeviceProperties(&prop, 0);

    printf("Device name: %s\n", prop.name);
    printf("Max threads per block: %d\n", prop.maxThreadsPerBlock);
    printf("Max threads per multiprocessor: %d\n", prop.maxThreadsPerMultiProcessor);
    printf("Warp size %d\n", prop.warpSize);
    printf("Max regs per block: %d\n", prop.regsPerBlock);
    printf("Max shared memory per block: %zu\n", prop.sharedMemPerBlock);
    printf("Major: %d\n", prop.major);
    printf("Minor: %d\n", prop.minor);
    printf("Max grid size: %d, %d, %d\n", prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
    printf("GCN arch: %d\n", prop.gcnArch);
    printf("GCN arch name: %s\n", prop.gcnArchName);
}