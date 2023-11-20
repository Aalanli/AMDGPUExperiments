# Sources

Compile on local machine without AMD gpu
[source](https://sep5.readthedocs.io/en/latest/Installation_Guide/Installation-Guide.html)
```bash
wget -qO - http://repo.radeon.com/rocm/apt/debian/rocm.gpg.key | sudo apt-key add -
echo 'deb [arch=amd64] http://repo.radeon.com/rocm/apt/debian/ xenial main' | sudo tee /etc/apt/sources.list.d/rocm.list

sudo apt update
sudo apt install rocm-dev
```

print configuration
```bash
hipconfig
```

compile for nvidia
```bash
export HIP_PLATFORM=nvidia
hipcc test.cpp ...
```

## Matrix Instructions
[AMD Lab notes](https://gpuopen.com/learn/amd-lab-notes/amd-lab-notes-matrix-cores-readme/)
[Matrix Instruction Calculator](https://github.com/RadeonOpenCompute/amd_matrix_instruction_calculator)
> $ ./matrix_calculator.py --architecture cdna2 --instruction v_mfma_f32_16x16x16f16 --register-layout --D-matrix
> d = __builtin_amdgcn_mfma_CDFmt_MxNxKABFmt (a, b, c, cbsz, abid, blgp)

## Profiler
https://gpuopen.com/learn/amd-lab-notes/amd-lab-notes-profilers-readme/

- Omnitrace => NsightSystems
- Omniperf => NsightCompute
  - https://amdresearch.github.io/omniperf/installation.html

| instruction            | blocks | regs per warp (a/b) | regs per warp (c/d) | CBSZ/ABID | BLGP  |
| ---------------------  | ------ | ------------------- | ------------------- | --------- | ----- |
| v_mfma_f32_32x32x1f32  | 2      | 1                   | 32                  | true      | true  |
| v_mfma_f32_16x16x1f32  | 4      | 1                   | 16                  | true      | true  |
| v_mfma_f32_4x4x1f32    | 16     | 1                   | 4                   | true      | true  |
| v_mfma_f32_32x32x2f32  | 1      | 1                   | 16                  | false     | true  |
| v_mfma_f32_16x16x4f32  | 1      | 1                   | 4                   | false     | true  |
| v_mfma_f32_32x32x4f16  | 2      | 2                   | 32                  | true      | true  |
| v_mfma_f32_16x16x4f16  | 4      | 2                   | 16                  | true      | true  |
| v_mfma_f32_4x4x4f16    | 16     | 2                   | 4                   | true      | true  |
| v_mfma_f32_32x32x8f16  | 1      | 2                   | 16                  | false     | true  |
| v_mfma_f32_16x16x16f16 | 1      | 2                   | 4                   | false     | true  |

### Omniperf

**Install**

```bash
tar xfz omniperf-v1.1.0-PR1.tar.gz
cd omniperf-v1.1.0-PR1
export INSTALL_DIR=~/omniperf

pip install -r requirements.txt
mkdir build
cd build
cmake -DCMAKE_INSTALL_PREFIX=${INSTALL_DIR}/1.1.0-PR1 \
        -DPYTHON_DEPS=${INSTALL_DIR}/python-libs \
        -DMOD_INSTALL_PATH=${INSTALL_DIR}/modulefiles ..

make install
```

**Profile**
```bash
omniperf profile -n vcopy_data -- ./vcopy
```
*It seems like profiling a python script does not work*


**Visualize**
```bash
omniperf analyze -p workloads/vcopy/mi200/ --gui
```

## ISA
[AMD_GCN](https://gpuopen.com/wp-content/uploads/2016/08/AMD_GCN3_Instruction_Set_Architecture_rev1.1.pdf)
[amdgcn-assembly](https://gpuopen.com/learn/amdgcn-assembly/)
```

                  cap gfx000 gfx803 gfx900 gfx906 gfx908 gfx90a gfx940 gfx941 gfx942 gfx1010 gfx1011 gfx1012 gfx1030 gfx1031 gfx1100 gfx1101 gfx1102 
      HasMFMA_bf16_1k      0      0      0      0      0      1      1      0      0       0       0       0       0       0       0       0       0 
       HasMFMA_i8_908      0      0      0      0      1      1      0      0      0       0       0       0       0       0       0       0       0 
       HasMFMA_i8_940      0      0      0      0      0      0      1      0      0       0       0       0       0       0       0       0       0 
           HasAddLshl      0      0      1      1      1      1      1      0      0       1       1       1       1       1       1       1       1 
         HasAtomicAdd      0      0      0      0      1      1      1      0      0       0       0       0       0       0       1       1       1 
   HasDirectToLdsDest      0      0      0      0      0      0      0      0      0       0       0       0       0       0       0       0       0 
 HasDirectToLdsNoDest      0      1      1      1      1      1      1      0      0       1       1       1       1       1       0       0       0 
        HasExplicitCO      0      0      1      1      1      1      1      0      0       1       1       1       1       1       1       1       1 
        HasExplicitNC      0      0      0      0      0      0      0      0      0       1       1       1       1       1       1       1       1 
       HasGLCModifier      0      1      1      1      1      1      0      0      0       1       1       1       1       1       1       1       1 
            HasLshlOr      0      0      1      1      1      1      1      0      0       1       1       1       1       1       1       1       1 
              HasMFMA      0      0      0      0      1      1      1      0      0       0       0       0       0       0       0       0       0 
            HasSMulHi      0      0      1      1      1      1      1      0      0       1       1       1       1       1       1       1       1 
              HasWMMA      0      0      0      0      0      0      0      0      0       0       0       0       0       0       1       1       1 
           MaxLgkmcnt      1      1      1      1      1      1      1      1      1       1       1       1       1       1       1       1       1 
             MaxVmcnt      0      1      1      1      1      1      1      0      0       1       1       1       1       1       1       1       1 
         SupportedISA      0      1      1      1      1      1      1      0      0       1       1       1       1       1       1       1       1 
      SupportedSource      1      1      1      1      1      1      1      1      1       1       1       1       1       1       1       1       1 
           HasMFMA_b8      0      0      0      0      0      0      1      0      0       0       0       0       0       0       0       0       0 
     HasMFMA_constSrc      0      0      0      0      0      1      1      0      0       0       0       0       0       0       0       0       0 
       v_dot2_f32_f16      0      0      0      1      1      1      1      0      0       0       1       1       1       1       1       1       1 
      v_dot2c_f32_f16      0      0      0      0      1      1      1      0      0       0       1       1       1       1       1       1       1 
            v_fma_f16      0      0      1      1      1      1      1      0      0       1       1       1       1       1       1       1       1 
           v_fmac_f16      0      0      0      0      0      0      0      0      0       0       0       0       0       0       0       0       0 
            v_mac_f16      0      1      1      1      1      1      1      0      0       0       0       0       0       0       0       0       0 
         v_pk_fma_f16      0      0      1      1      1      1      1      0      0       1       1       1       1       1       1       1       1 
        v_pk_fmac_f16      0      0      0      0      0      0      0      0      0       0       0       0       0       0       0       0       0 
            v_fma_f32      0      1      1      1      1      1      1      0      0       1       1       1       1       1       1       1       1 
        v_fma_mix_f32      0      0      0      1      1      1      1      0      0       1       1       1       1       1       1       1       1 
           v_fmac_f32      0      0      0      1      1      1      1      0      0       1       1       1       1       1       1       1       1 
            v_mac_f32      0      1      1      1      1      1      0      0      0       1       1       1       0       0       0       0       0 
        v_mad_mix_f32      0      0      1      0      0      0      0      0      0       0       0       0       0       0       0       0       0 
          HasMFMA_f64      0      0      0      0      0      1      1      0      0       0       0       0       0       0       0       0       0 
            v_fma_f64      0      1      1      1      1      1      1      0      0       1       1       1       1       1       1       1       1 
           HasMFMA_f8      0      0      0      0      0      0      1      0      0       0       0       0       0       0       0       0       0 
    VOP3v_dot4_i32_i8      0      0      0      1      1      1      1      0      0       0       1       1       1       1       0       0       0 
        v_dot4_i32_i8      0      0      0      0      0      0      0      0      0       0       0       0       0       0       0       0       0 
       v_dot4c_i32_i8      0      0      0      0      1      1      1      0      0       0       1       1       1       1       0       0       0 
HasMFMA_bf16_original      0      0      0      0      1      1      0      0      0       0       0       0       0       0       0       0       0 
         HasMFMA_vgpr      0      0      0      0      0      1      1      0      0       0       0       0       0       0       0       0       0 
         HasMFMA_xf32      0      0      0      0      0      0      1      0      0       0       0       0       0       0       0       0       0 
   ArchAccUnifiedRegs      0      0      0      0      0      1      1      1      1       0       0       0       0       0       0       0       0 
       CMPXWritesSGPR      1      1      1      1      1      1      1      1      1       0       0       0       0       0       0       0       0 
        CrosslaneWait      0      0      0      0      0      0      1      1      1       0       0       0       0       0       0       0       0 
        ForceStoreSC1      0      0      0      0      0      0      1      1      0       0       0       0       0       0       0       0       0 
             HasAccCD      0      0      0      0      0      1      1      1      1       0       0       0       0       0       0       0       0 
           HasEccHalf      0      0      0      1      1      1      1      1      1       0       0       0       0       0       0       0       0 
            HasWave32      0      0      0      0      0      0      0      0      0       1       1       1       1       1       1       1       1 
           InstRename      0      0      0      0      0      0      0      0      0       0       0       0       0       0       1       1       1 
        SeparateVscnt      0      0      0      0      0      0      0      0      0       1       1       1       1       1       1       1       1 
             VgprBank      0      0      0      0      0      0      0      0      0       1       1       1       1       1       1       1       1 
     Waitcnt0Disabled      0      0      0      0      1      1      1      1      1       0       0       0       0       0       0       0       0
```

[Inline Assembly Example](https://github.com/ROCm-Developer-Tools/HIP/tree/master/samples/2_Cookbook/10_inline_asm)
[Inline Assembly Kernel Example](https://github.com/ROCm-Developer-Tools/LLVM-AMDGPU-Assembler-Extra/blob/master/examples/gfx8/ds_bpermute.s)

```asm
 asm volatile ("v_mov_b32_e32 %0, %1" : "=v" (out[x*width + y]) : "v" (in[y*width + x]));
```

v refers to VGPR register
=v refers to assignment of this register

export KMDUMPLLVM=1 : This gets you LLVM IR.
export KMDUMPISA=1 : This gets you GCN ISA.

[Assembly Crosslane ops](https://gpuopen.com/learn/amd-gcn-assembly-cross-lane-operations/)
```asm
ds_permute_b32 dest, addr, src [offset:addr_offset] // push to dest
ds_bpermute_b32 dest, addr, src [offset:addr_offset] // pull from src
```

Haha apparently cuda equivalents are in `#include <hip/amd_detail/amd_warp_functions.h>`
So no need to write inline assembly to replicate cuda warp shuffles

### Interesting Instructions used in ROCBlas
[CDNA2 ISA Manual](https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/instruction-set-architectures/instinct-mi200-cdna2-instruction-set-architecture.pdf)

```asm
/******************************************/
/* 2x2 thread-tile                        */
/******************************************/
.macro MAC_2x2_X0
// Component.MAC.MAC_I8X4_Plain
v_dot4_i32_i8 v[vgprValuC+0+0*2], v[vgprValuA_X0_I0+0], v[vgprValuB_X0_I0+0], v[vgprValuC+0+0*2] op_sel:[0,0] op_sel_hi:[1,1] //valuC[0]
s_setprio 1 // Raise priority while processing macs
v_dot4_i32_i8 v[vgprValuC+1+0*2], v[vgprValuA_X0_I0+1], v[vgprValuB_X0_I0+0], v[vgprValuC+1+0*2] op_sel:[0,0] op_sel_hi:[1,1] //valuC[1]
v_dot4_i32_i8 v[vgprValuC+0+1*2], v[vgprValuA_X0_I0+0], v[vgprValuB_X0_I0+1], v[vgprValuC+0+1*2] op_sel:[0,0] op_sel_hi:[1,1] //valuC[2]
v_dot4_i32_i8 v[vgprValuC+1+1*2], v[vgprValuA_X0_I0+1], v[vgprValuB_X0_I0+1], v[vgprValuC+1+1*2] op_sel:[0,0] op_sel_hi:[1,1] //valuC[3]
s_setprio 0 // Reset priority after macs
.endm
.macro MAC_2x2_X1
// Component.MAC.MAC_I8X4_Plain
v_dot4_i32_i8 v[vgprValuC+0+0*2], v[vgprValuA_X1_I0+0], v[vgprValuB_X1_I0+0], v[vgprValuC+0+0*2] op_sel:[0,0] op_sel_hi:[1,1] //valuC[0]
s_setprio 1 // Raise priority while processing macs
v_dot4_i32_i8 v[vgprValuC+1+0*2], v[vgprValuA_X1_I0+1], v[vgprValuB_X1_I0+0], v[vgprValuC+1+0*2] op_sel:[0,0] op_sel_hi:[1,1] //valuC[1]
v_dot4_i32_i8 v[vgprValuC+0+1*2], v[vgprValuA_X1_I0+0], v[vgprValuB_X1_I0+1], v[vgprValuC+0+1*2] op_sel:[0,0] op_sel_hi:[1,1] //valuC[2]
v_dot4_i32_i8 v[vgprValuC+1+1*2], v[vgprValuA_X1_I0+1], v[vgprValuB_X1_I0+1], v[vgprValuC+1+1*2] op_sel:[0,0] op_sel_hi:[1,1] //valuC[3]
s_setprio 0 // Reset priority after macs
.endm
```

> v_dot4_i32_i8
```
D.i32 = S0.i8[0] * S1.i8[0] + S0.i8[1] * S1.i8[1] + S0.i8[2] *
S1.i8[2] + S0.i8[3] * S1.i8[3] + S2.i32
```

### MI200 ISA Notes
scalar alu instructions take up to 2 s_gprs, or literal constants
vector alu instructions take up to three arguments, vgprs, sgprs, or literal constants
vector memory instructions transfer data between vgprs and memory

#### Program State
PC - Program Counter (48 bits)
V0-V255 - 256 VGPRs (32 bits)
AV0-AV255 - 256 Matrix accumulation VGPRs (32 bits)
- vgprs are allocated in 16, inst on 64 bit must be even aligned
S0-S103 - 104 SGPRs (32 bits)
- max 102 available to a wavefront
LDS - 64kB
EXEC - 64 bits
VCC - Bit mask holds result from vector compare (64 bits)

FLAT_SCRATCH - base address of scratch memory
M0 - Memory Reg (32 bits)

VMCNT - Counts VMEM issued instructions but not completed (6 bits)
EXPCNT - Counts GDS issued (3 bits)
LGKMCNT - LDS, GDS, Constant, Message (4 bits)

SCC - result from scalar alu compare (1 bit)

- execute mask affects vector alu, vector memory, lds, gds instructions

#### Data Dependency Resolution
Instructions fo same type return in order, different types return out of order

S_WAITCNT waits for issued instruction counters to be below a value

VM_CNT: vector memory count 
- incremented a memory read or write is issued (MIMG, MUBUF, MTBUF)
- decremented when completed

LGKM_CNT (LDS, GDS, K(constant), (M)essage): when low latency instructions are completed

##### Some common manual NOPs
VALU that sets VCC or EXEC -> VALU that uses EXECZ or VCCZ : 5 wait

#### Scalar Instructions
Compare instructions sets SCC, select and jump reads SCC as the conditional
- does not appear to be any data resolution issues for reading/writing SCC

#### Vector ALU Instructions
VALU instructions have 4 encodings, VOP3 uses 64 bits, and 3 other 32 bit encodings
use 32 bit encoding whenever possible

- vgpr ACC registers can be used for A/B matrix, and must be used for C/D matrix

#### Scalar Mem Instructions
- mostly for loading constants, increments LGKM_CNT
- can read 1 to 16 dwords, write 1 to 4 dwords into sgprs through scalar data cache

#### Vector Mem Instructions
Vector memory instructions go through l1 and l2 caches, called texture cache
There are three types of loads/stores/atomics
- MTBUF: memory typed-buffer
  - data format specified in instruction
  - load,store
- MUBUF: memory untyped-buffer
  - data format specified in descriptor
  - load,store,atomic
- MIMG memory image

Instructions defines vgpr(s) for the address, vgprs for the dest/src, and sgprs for buffer descriptor

- buffer reads have the option of returning data to vgprs or directly into lds
- buffer resource constant/buffer descriptor describes address and characteristics of buffer in memory

Flat instructions do not need resource descriptors, treating memory as flat

#### LDS
64 kB, 32 banks of 512 dwords
reads across wavefront executed in 4 cycles

LDS direct reads, when all threads access the same dword address, has no bank conflicts

[builtin_amdgcn list](https://github.com/llvm-mirror/clang/blob/master/include/clang/Basic/BuiltinsAMDGPU.def)
eg: `__builtin_amdgcn_ds_gws_barrier`

#### MISC
output gcn asm
```bash
hipcc test.cpp -save-temps
```

## Composable Kernel

Compiler does not generate MUBUF/MTBUF instruction, but `global_load` flat buffer instructions

Use this to force buffer loads, contained in `ck/utility/amd_buffer_addressing.hpp:145`
```c++
// buffer load fp32
__device__ float
llvm_amdgcn_raw_buffer_load_fp32(int32x4_t srsrc,
                                 index_t voffset,
                                 index_t soffset,
                                 index_t glc_slc) __asm("llvm.amdgcn.raw.buffer.load.f32");

__device__ float2_t
llvm_amdgcn_raw_buffer_load_fp32x2(int32x4_t srsrc,
                                   index_t voffset,
                                   index_t soffset,
                                   index_t glc_slc) __asm("llvm.amdgcn.raw.buffer.load.v2f32");

__device__ float4_t
llvm_amdgcn_raw_buffer_load_fp32x4(int32x4_t srsrc,
                                   index_t voffset,
                                   index_t soffset,
                                   index_t glc_slc) __asm("llvm.amdgcn.raw.buffer.load.v4f32");
```
Where `srsrc` is the 128 bit buffer descriptor stored in sgprs
srsrc[63:0] = addr of pointer
srsrc[95:64] = element_space_size or 0xffffffff
srsrc[127:96] = 0x00020000 
See isa manual page 66, table 36 "buffer resource descriptor" for more details.

[Mi200 ISA](https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/instruction-set-architectures/instinct-mi200-cdna2-instruction-set-architecture.pdf)
Page 74:
> The MUBUF instruction format allows reading data from a memory buffer directly into LDS
without passing through VGPRs

## MISC
**Torch HIP Semantics**
https://pytorch.org/docs/stable/notes/hip.html

[mix-bench](https://github.com/ekondis/mixbench)
[HIP](https://github.com/ROCm-Developer-Tools/HIP)
