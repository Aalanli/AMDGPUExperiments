project(GPUExperiments)

cmake_minimum_required(VERSION 3.16)

# Search for rocm in common locations
list(APPEND CMAKE_PREFIX_PATH /opt/rocm/hip /opt/rocm /opt/rocm/rocblas)
# list(APPEND CUDA_NVCC_FLAGS "--compiler-options -fPIC")

# Find HIP.
# The user may override AMDGPU_TARGETS defined in the HIP config file
# to select the AMDGPU archs to compile for.
# ex. set(AMDGPU_TARGETS "gfx803;gfx900;gfx906")
find_package(hip REQUIRED)
find_package(rocblas REQUIRED)


# Find OpenMP.
# find_package(OpenMP REQUIRED)

# Set compiler and linker.
set(CMAKE_CXX_COMPILER ${HIP_HIPCC_EXECUTABLE})
set(CMAKE_CXX_LINKER   ${HIP_HIPCC_EXECUTABLE})

set(CMAKE_BUILD_TYPE Release)


# Preparing the executable.
add_executable(helloworld
    ${CMAKE_SOURCE_DIR}/src/helloworld.cpp
    ${CMAKE_SOURCE_DIR}/src/saxpy.cpp
)
# Link Libraries - HIP Device and OpenMP.
target_include_directories(helloworld PRIVATE include)
target_compile_options(helloworld PRIVATE ${OpenMP_CXX_FLAGS})
target_link_libraries(helloworld PRIVATE hip::device ${OpenMP_CXX_FLAGS})


add_executable(gemm
    ${CMAKE_SOURCE_DIR}/src/gemm.cpp
)

target_include_directories(gemm PRIVATE include)
target_compile_options(gemm PRIVATE ${OpenMP_CXX_FLAGS})
target_link_libraries(gemm PRIVATE hip::device roc::rocblas ${OpenMP_CXX_FLAGS})


add_executable(copy
    ${CMAKE_SOURCE_DIR}/src/copy.cpp
)

target_include_directories(copy PRIVATE include)
target_compile_options(copy PRIVATE ${OpenMP_CXX_FLAGS})
target_link_libraries(copy PRIVATE hip::device ${OpenMP_CXX_FLAGS})


# add_library(hello SHARED
#     ${CMAKE_SOURCE_DIR}/src/test.cpp
# )

# target_include_directories(hello PRIVATE include)
# target_compile_options(hello PRIVATE ${OpenMP_CXX_FLAGS})
# target_link_libraries(hello PRIVATE hip::device ${OpenMP_CXX_FLAGS})

# hipcc -fPIC -O3 -c hello.cpp
# hipcc -shared -o libhello.so hello.o
# nm -D libhello.so