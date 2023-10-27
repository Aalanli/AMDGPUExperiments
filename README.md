# Sources

Compile on local machine without AMD gpu
[source](https://sep5.readthedocs.io/en/latest/Installation_Guide/Installation-Guide.html)
```bash
wget -qO - http://repo.radeon.com/rocm/apt/debian/rocm.gpg.key | sudo apt-key add -
echo 'deb [arch=amd64] http://repo.radeon.com/rocm/apt/debian/ xenial main' | sudo tee /etc/apt/sources.list.d/rocm.list

sudo apt update
sudo apt install rocm-dev
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
## MISC
**Torch HIP Semantics**
https://pytorch.org/docs/stable/notes/hip.html

[mix-bench](https://github.com/ekondis/mixbench)
[HIP](https://github.com/ROCm-Developer-Tools/HIP)
