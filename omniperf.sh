omniperf profile -n gemm -- ./omniperf.sh
wait
omniperf analyze -p workloads/gemm/mi200/ --gui