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
  

## MISC
**Torch HIP Semantics**
https://pytorch.org/docs/stable/notes/hip.html

[mix-bench](https://github.com/ekondis/mixbench)
[HIP](https://github.com/ROCm-Developer-Tools/HIP)
