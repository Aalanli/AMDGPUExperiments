cmake_minimum_required(VERSION 3.14)
project(GPUExperiments)
add_compile_options(-std=c++17 -Wno-unused-result)

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

function(BUILD_THIS FPATH SAVETEMPS)
    get_filename_component(FNAME ${FPATH} NAME_WE)
    add_library(
        ${FNAME} SHARED
        ${CMAKE_SOURCE_DIR}/src/${FPATH}
        ${CMAKE_SOURCE_DIR}/src/utils.cpp
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
    ${CMAKE_SOURCE_DIR}/src/ck/composable_kernel.cpp
)

target_include_directories(ck_test PRIVATE include)
target_compile_options(ck_test PRIVATE ${OpenMP_CXX_FLAGS})
target_link_libraries(ck_test PRIVATE hip::device composable_kernel::device_operations ${OpenMP_CXX_FLAGS})


# add_executable(wmma_test
#     ${CMAKE_SOURCE_DIR}/src/misc/rocwmma_example.cpp
# )

# target_compile_options(wmma_test PRIVATE ${OpenMP_CXX_FLAGS})
# target_link_libraries(wmma_test PRIVATE hip::device ${OpenMP_CXX_FLAGS})


add_library(
    ck_gemm SHARED
    ${CMAKE_SOURCE_DIR}/src/ck/composable_kernel_client.cpp    
)

target_include_directories(ck_gemm PRIVATE include)
target_compile_options(ck_gemm PRIVATE ${OpenMP_CXX_FLAGS})
target_link_libraries(ck_gemm PRIVATE hip::device composable_kernel::device_operations ${OpenMP_CXX_FLAGS})

add_library(
    ck_gemm_dl SHARED
    ${CMAKE_SOURCE_DIR}/src/ck/ck_kernel_f32_dl_client.cpp    
)

target_include_directories(ck_gemm_dl PRIVATE include)
target_compile_options(ck_gemm_dl PRIVATE ${OpenMP_CXX_FLAGS})
target_link_libraries(ck_gemm_dl PRIVATE hip::device composable_kernel::device_operations ${OpenMP_CXX_FLAGS})

add_executable(gemm
    ${CMAKE_SOURCE_DIR}/src/misc/gemm.cpp
)

target_include_directories(gemm PRIVATE include)
target_compile_options(gemm PRIVATE ${OpenMP_CXX_FLAGS})
target_link_libraries(gemm PRIVATE hip::device roc::rocblas ${OpenMP_CXX_FLAGS})


add_executable(mfma_tools_test
    ${CMAKE_SOURCE_DIR}/src/mma_gemm/mfma_tools_test.cpp
)


target_include_directories(mfma_tools_test PRIVATE include)
target_compile_options(mfma_tools_test PRIVATE ${OpenMP_CXX_FLAGS})
target_link_libraries(mfma_tools_test PRIVATE hip::device ${OpenMP_CXX_FLAGS})


build_this("utils.cpp" OFF)
build_this("misc/helloworld.cpp" OFF)
build_this("misc/saxpy.cpp" OFF)
build_this("misc/rocblas_gemm.cpp" OFF)
build_this("misc/ds_perm.cpp" OFF)
build_this("misc/pipeline_test.cpp" OFF)
build_this("misc/copy.cpp" OFF)
build_this("simt_gemm/simt_gemm.cpp" OFF)
build_this("simt_gemm/simt_gemm_hidet.cpp" OFF)
build_this("mma_gemm/mfma_gemm.cpp" OFF)
build_this("mma_gemm/mfma_tools.cpp" OFF)
build_this("mma_gemm/mfma_gemmv2.cpp" OFF)
build_this("mma_gemm/mfma_gemmv3.cpp" OFF)
# build_this("mma_gemm/rocwmma_gemm.cpp" OFF)
build_this("mma_gemm/layouts.cpp" OFF)
build_this("mma_gemm/warp_tiles.cpp" OFF)
build_this("mma_gemm/block_tiles.cpp" OFF)
build_this("mma_gemm/kernel_tiles.cpp" OFF)
