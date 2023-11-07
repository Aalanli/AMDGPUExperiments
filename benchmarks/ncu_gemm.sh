python benchmarks/gemm_ncu.py
ncu -f -o benchmarks/gemm_simt --set full python benchmarks/gemm_ncu.py
ncu-ui benchmarks/gemm_simt.ncu-rep