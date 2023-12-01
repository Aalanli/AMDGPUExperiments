#include "hip_utils.hpp"


#if defined(__gfx803__) || defined(__gfx900__) || defined(__gfx906__) || defined(__gfx908__) || \
    defined(__gfx90a__) || defined(__gfx940__) || defined(__gfx941__) ||                          \
    defined(__gfx942__) // for GPU code
#define CK_BUFFER_RESOURCE_3RD_DWORD 0x00020000
#elif defined(__gfx1030__) // for GPU code
#define CK_BUFFER_RESOURCE_3RD_DWORD 0x31014000
#elif defined(__gfx1100__) || defined(__gfx1101__) || defined(__gfx1102__) // for GPU code
#define CK_BUFFER_RESOURCE_3RD_DWORD 0x31004000
#else
#define CK_BUFFER_RESOURCE_3RD_DWORD 0x00020000
#endif


template <typename T>
union BufferResource
{
    __device__ constexpr BufferResource() : content{} {}

    // 128 bit SGPRs to supply buffer resource in buffer instructions
    // https://rocm-documentation.readthedocs.io/en/latest/GCN_ISA_Manuals/testdocbook.html#vector-memory-buffer-instructions
    int32x4_t content;
    T* address[2];
    int range[4];
    int config[4];
};


template <typename T>
__device__ int32x4_t make_wave_buffer_resource(T* p_wave, int element_space_size)
{
    BufferResource<T> wave_buffer_resource;

    // wavewise base address (64 bit)
    wave_buffer_resource.address[0] = const_cast<T*>(p_wave);
    // wavewise range (32 bit)
    wave_buffer_resource.range[2] = element_space_size * sizeof(T);
    // wavewise setting (32 bit)
    wave_buffer_resource.config[3] = CK_BUFFER_RESOURCE_3RD_DWORD;

    return wave_buffer_resource.content;
}


// buffer load i8
__device__ int8_t
llvm_amdgcn_raw_buffer_load_i8(int32x4_t srsrc,
                               int voffset,
                               int soffset,
                               int glc_slc) __asm("llvm.amdgcn.raw.buffer.load.i8");

// buffer load i16
__device__ ushort
llvm_amdgcn_raw_buffer_load_i16(int32x4_t srsrc,
                                int voffset,
                                int soffset,
                                int glc_slc) __asm("llvm.amdgcn.raw.buffer.load.i16");


__device__ int
llvm_amdgcn_raw_buffer_load_i32(int32x4_t srsrc,
                                int voffset,
                                int soffset,
                                int glc_slc) __asm("llvm.amdgcn.raw.buffer.load.i32");

__device__ int32x2_t
llvm_amdgcn_raw_buffer_load_i32x2(int32x4_t srsrc,
                                  int voffset,
                                  int soffset,
                                  int glc_slc) __asm("llvm.amdgcn.raw.buffer.load.v2i32");

__device__ int32x4_t
llvm_amdgcn_raw_buffer_load_i32x4(int32x4_t srsrc,
                                  int voffset,
                                  int soffset,
                                  int glc_slc) __asm("llvm.amdgcn.raw.buffer.load.v4i32");


enum struct AmdBufferCoherenceEnum
{
    DefaultCoherence = 0, // default value
    GLC              = 1,
    SLC              = 2,
    GLC_SLC          = 3,
};

template <int N, AmdBufferCoherenceEnum coherence = AmdBufferCoherenceEnum::DefaultCoherence>
__device__ auto
amd_buffer_load_impl_raw(int32x4_t src_wave_buffer_resource,
                         int src_thread_addr_offset,
                         int src_wave_addr_offset)
{
    static_assert(N == 1 || N == 2 || N == 4 || N == 8 || N == 16,
                  "wrong! not implemented");

    if constexpr(N == 1)
    {
        return llvm_amdgcn_raw_buffer_load_i8(src_wave_buffer_resource,
                                              src_thread_addr_offset,
                                              src_wave_addr_offset,
                                              static_cast<int>(coherence));
    }
    else if constexpr(N == 2)
    {

        return llvm_amdgcn_raw_buffer_load_i16(src_wave_buffer_resource,
                                                      src_thread_addr_offset,
                                                      src_wave_addr_offset,
                                                      static_cast<int>(coherence));
    }
    else if constexpr(N == 4)
    {
        return llvm_amdgcn_raw_buffer_load_i32(src_wave_buffer_resource,
                                                      src_thread_addr_offset,
                                                      src_wave_addr_offset,
                                                      static_cast<int>(coherence));

    }
    else if constexpr(N == 8)
    {
        return llvm_amdgcn_raw_buffer_load_i32x2(src_wave_buffer_resource,
                                                          src_thread_addr_offset,
                                                          src_wave_addr_offset,
                                                          static_cast<int>(coherence));

    }
    else if constexpr(N == 16)
    {
        return llvm_amdgcn_raw_buffer_load_i32x4(src_wave_buffer_resource,
                                                          src_thread_addr_offset,
                                                          src_wave_addr_offset,
                                                          static_cast<int>(coherence));
    }
}

// buffer_load requires:
//   1) p_src_wave must point to global memory space
//   2) p_src_wave must be a wavewise pointer.
// It is user's responsibility to make sure that is true.
// each thread then writes to p_dest_wave, local registers
template <typename T,
          int N,
          AmdBufferCoherenceEnum coherence = AmdBufferCoherenceEnum::DefaultCoherence>
__device__ inline void amd_buffer_load_invalid_element_set_zero(
    const T* p_src_wave,
    int src_thread_element_offset,
    bool src_thread_element_valid,
    int src_element_space_size,
    T* p_dest_wave)
{
    const int32x4_t src_wave_buffer_resource =
        make_wave_buffer_resource(p_src_wave, src_element_space_size);

    int src_thread_addr_offset = src_thread_element_offset * sizeof(T);
    auto tmp = amd_buffer_load_impl_raw<sizeof(T) * N>(
        src_wave_buffer_resource, src_thread_addr_offset, 0);

    if (src_thread_element_valid) {
        if constexpr(sizeof(T) == 4) {
            #pragma unroll
            for (int i = 0; i < sizeof(T) * N / 4; ++i) {
                ((int*) p_dest_wave)[i] = ((int*) &tmp)[i];
            }    
        } else {
            // hopefully the compiler will auto vectorize this
            #pragma unroll
            for (int i = 0; i < sizeof(T) * N; ++i) {
                ((int8_t*) p_dest_wave)[i] = ((int8_t*) &tmp)[i];
            }
        }
    } else {
        #pragma unroll
        for (int i = 0; i < N; ++i) {
            p_dest_wave[i] = T{};
        }
    }
}