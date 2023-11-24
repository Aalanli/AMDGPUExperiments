# %%
import torch
from kernels.gemm import rocgemm
from kernels.gemm_hidet import hidet_simt
from kernels.composable_kernel_gemm import ck_gemm, ck_gemm_dl
from kernels.gemm_mfmav1 import mfma_gemmv1
from kernels.gemm_mfmav2 import mfma_gemmv2
from kernels.gemm_mfmav3 import mfma_gemmv3
from kernels.rocwmma_gemm import wmma_gemm
from kernels.utils import Bench
from triton.ops.matmul import matmul

if __name__ == '__main__':
    def benchmark_func(f):
        def bench_fn(i, **kwargs):
            m, k, n = i
            a = torch.randn([m, k], device='cuda')
            b = torch.randn([k, n], device='cuda')
            return lambda: f(a, b)
        return bench_fn
    
    square = [(m, m, m) for m in range(256, 2049, 256)]
    mem_bound = [(1, m, m) for m in [32, 64, 128, 256, 512, 768, 1024]]
    bench = Bench(
        x_vals=square,
        x_name='(m, k, n)',
    )
    # bench.bench(bench_rocgemm1)
    # bench.bench(bench_rocgemm2, 'naive')
    bench.bench(benchmark_func(lambda a, b: a @ b), 'blas')
    # bench.bench(benchmark_func(hidet_simt),   'simt')
    # bench.bench(benchmark_func(lambda a, b: hidet_simt(a, b, version=1)), 'simtv2')
    # bench.bench(benchmark_func(lambda a, b: hidet_simt(a, b, version=2)), 'simtv3')
    # bench.bench(benchmark_func(ck_gemm), "composable_kernel_mfma")
    # bench.bench(benchmark_func(ck_gemm_dl), "composable_kernel_simt")
    # bench.bench(benchmark_func(mfma_gemmv1), "mfma_v1")
    bench.bench(benchmark_func(lambda a, b: mfma_gemmv1(a, b, version=1)), "mfma_v2")
    bench.bench(benchmark_func(wmma_gemm), "wmma_v1")
    # bench.bench(benchmark_func(mfma_gemmv2), "wmma_mfma")
    # bench.bench(benchmark_func(lambda a, b: mfma_gemmv2(a, b, ver=1, pack_len=4)), "wmma_mfma_v1_pack4")
    # bench.bench(benchmark_func(lambda a, b: mfma_gemmv2(a, b, ver=1, pack_len=2)), "wmma_mfma_v1_pack2")
    bench.bench(benchmark_func(mfma_gemmv3), "mfma_v3")
    
    # bench.bench(benchmark_func(rocgemm), "gemm naive")
    # bench.bench(benchmark_func(matmul), "triton")
    data = bench.run()
    data.show_plot()
    data.print_data()

#                         blas      simt    simtv2    simtv3  \
# (256, 256, 256)     0.017440  0.037280  0.037760  0.037120   
# (512, 512, 512)     0.026720  0.070721  0.068000  0.071840   
# (768, 768, 768)     0.043840  0.141760  0.122560  0.130240   
# (1024, 1024, 1024)  0.083040  0.236961  0.232481  0.254720   
# (1280, 1280, 1280)  0.154720  0.318880  0.307361  0.326400   
# (1536, 1536, 1536)  0.254081  0.558240  0.530240  0.588480   
# (1792, 1792, 1792)  0.386560  0.859681  0.824161  0.884161   
# (2048, 2048, 2048)  0.522721  1.094081  1.027841  1.139841   

#                     composable_kernel_mfma  composable_kernel_simt  
# (256, 256, 256)                   0.030720                0.065600  
# (512, 512, 512)                   0.049600                0.117760  
# (768, 768, 768)                   0.069760                0.172960  
# (1024, 1024, 1024)                0.123680                0.232000  
# (1280, 1280, 1280)                0.160000                0.296321  
# (1536, 1536, 1536)                0.258240                0.597761  
# (1792, 1792, 1792)                0.381601                0.715841  
# (2048, 2048, 2048)                0.550560                1.174401  