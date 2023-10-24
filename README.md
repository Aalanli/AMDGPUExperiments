# Sources

## Matrix Instructions
[AMD Lab notes](https://gpuopen.com/learn/amd-lab-notes/amd-lab-notes-matrix-cores-readme/)
[Matrix Instruction Calculator](https://github.com/RadeonOpenCompute/amd_matrix_instruction_calculator)
> $ ./matrix_calculator.py --architecture cdna2 --instruction v_mfma_f32_16x16x16f16 --register-layout --D-matrix
> d = __builtin_amdgcn_mfma_CDFmt_MxNxKABFmt (a, b, c, cbsz, abid, blgp)

## ISA
[AMD_GCN](https://gpuopen.com/wp-content/uploads/2016/08/AMD_GCN3_Instruction_Set_Architecture_rev1.1.pdf)
[amdgcn-assembly](https://gpuopen.com/learn/amdgcn-assembly/)


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


## MISC
**Torch HIP Semantics**
https://pytorch.org/docs/stable/notes/hip.html

[mix-bench](https://github.com/ekondis/mixbench)
[HIP](https://github.com/ROCm-Developer-Tools/HIP)
