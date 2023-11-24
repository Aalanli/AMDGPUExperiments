omniperf profile -n gemm -- ./omniperf_launch.sh
wait
omniperf analyze -p workloads/gemm/mi200/ --gui