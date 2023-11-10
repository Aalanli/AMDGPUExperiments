omniperf profile -n gemm -- ./benchmarks/omniperf.sh
wait
omniperf analyze -p workloads/gemm/mi200/ --gui