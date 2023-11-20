project(GPUExperiments)
add_compile_options(-std=c++17 -save-temps)

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
find_package(composable_kernel COMPONENTS device_operations)

# set(GPU_TARGETS "gfx90a")
# Find OpenMP.
# find_package(OpenMP REQUIRED)

# Set compiler and linker.
set(CMAKE_CXX_COMPILER ${HIP_HIPCC_EXECUTABLE})
set(CMAKE_CXX_LINKER   ${HIP_HIPCC_EXECUTABLE})

set(CMAKE_BUILD_TYPE Release)

function(make_lib, FNAME, SAVETEMPS)
    add_library(
        ${FNAME} SHARED
        ${CMAKE_SOURCE_DIR}/src/${FNAME}.cpp
    )
    target_include_directories(${FNAME} PRIVATE include)
    if(${SAVETEMPS})
        add_compile_options(${FNAME} PRIVATE ${OpenMP_CXX_FLAGS} -save-temps)
    else()
        target_compile_options(${FNAME} PRIVATE ${OpenMP_CXX_FLAGS})
    endif()
    target_link_libraries(${FNAME} PRIVATE hip::device ${OpenMP_CXX_FLAGS})
endfunction()

add_executable(ck_test
    ${CMAKE_SOURCE_DIR}/src/composable_kernel.cpp
)

target_include_directories(ck_test PRIVATE include)
target_compile_options(ck_test PRIVATE ${OpenMP_CXX_FLAGS})
target_link_libraries(ck_test PRIVATE hip::device composable_kernel::device_operations ${OpenMP_CXX_FLAGS})


add_executable(wmma_test
    ${CMAKE_SOURCE_DIR}/src/rocwmma_example.cpp
)

target_compile_options(wmma_test PRIVATE ${OpenMP_CXX_FLAGS})
target_link_libraries(wmma_test PRIVATE hip::device ${OpenMP_CXX_FLAGS})


make_lib(rocwmma_gemm, 0)
# add_library(wmma_gemm
#     ${CMAKE_SOURCE_DIR}/src/rocwmma_gemm.cpp
# )

# target_include_directories(wmma_gemm PRIVATE include)
# target_compile_options(wmma_gemm PRIVATE ${OpenMP_CXX_FLAGS})
# target_link_libraries(wmma_gemm PRIVATE hip::device ${OpenMP_CXX_FLAGS})


add_library(
    ck_gemm SHARED
    ${CMAKE_SOURCE_DIR}/src/composable_kernel_client.cpp    
)

target_include_directories(ck_gemm PRIVATE include)
target_compile_options(ck_gemm PRIVATE ${OpenMP_CXX_FLAGS})
target_link_libraries(ck_gemm PRIVATE hip::device composable_kernel::device_operations ${OpenMP_CXX_FLAGS})


add_library(
    ck_gemm_dl SHARED
    ${CMAKE_SOURCE_DIR}/src/ck_kernel_f32_dl_client.cpp    
)

target_include_directories(ck_gemm_dl PRIVATE include)
target_compile_options(ck_gemm_dl PRIVATE ${OpenMP_CXX_FLAGS})
target_link_libraries(ck_gemm_dl PRIVATE hip::device composable_kernel::device_operations ${OpenMP_CXX_FLAGS})


# Preparing the executable.
add_executable(helloworld
    ${CMAKE_SOURCE_DIR}/src/helloworld.cpp
    ${CMAKE_SOURCE_DIR}/src/saxpy.cpp
    ${CMAKE_SOURCE_DIR}/src/rocblas_gemm.cpp
    ${CMAKE_SOURCE_DIR}/src/simt_gemm.cpp
    ${CMAKE_SOURCE_DIR}/src/simt_gemmv2.cu
    ${CMAKE_SOURCE_DIR}/src/ds_perm.cpp
    ${CMAKE_SOURCE_DIR}/src/pipeline_test.cpp
    ${CMAKE_SOURCE_DIR}/src/simt_gemm_hidet.cpp 
    ${CMAKE_SOURCE_DIR}/src/mfma_gemm.cpp
    # ${CMAKE_SOURCE_DIR}/src/test_void_call.cpp
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


add_library(mfma_test
    ${CMAKE_SOURCE_DIR}/src/mfma_gemmv2.cpp
)

target_include_directories(mfma_test PRIVATE include)
target_compile_options(mfma_test PRIVATE ${OpenMP_CXX_FLAGS})
target_link_libraries(mfma_test PRIVATE hip::device ${OpenMP_CXX_FLAGS})


# add_library(hello SHARED
#     ${CMAKE_SOURCE_DIR}/src/test.cpp
# )

# target_include_directories(hello PRIVATE include)
# target_compile_options(hello PRIVATE ${OpenMP_CXX_FLAGS})
# target_link_libraries(hello PRIVATE hip::device ${OpenMP_CXX_FLAGS})

# hipcc -fPIC -O3 -c hello.cpp
# hipcc -shared -o libhello.so hello.o
# nm -D libhello.so