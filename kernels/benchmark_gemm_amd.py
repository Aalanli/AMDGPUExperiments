# %%
import torch
from kernels.gemm import rocgemm
from kernels.gemm_hidet import hidet_simt
from kernels.composable_kernel_gemm import ck_gemm, ck_gemm_dl
from kernels.gemm_mfmav1 import mfma_gemmv1
from kernels.gemm_mfmav2 import mfma_gemmv2
from kernels.gemm_mfmav3 import mfma_gemmv3, mfma_gemmv3_5
from kernels.rocwmma_gemm import wmma_gemm
from kernels.utils import Bench
from triton.ops.matmul import matmul

if __name__ == '__main__':
    def benchmark_func(f):
        def bench_fn(i, **kwargs):
            m, k, n = i, i, i
            a = torch.randn([m, k], device='cuda')
            b = torch.randn([k, n], device='cuda')
            return lambda: f(a, b)
        return bench_fn
    
    square = [m for m in range(256, 2049, 128)]
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
    bench.bench(benchmark_func(ck_gemm), "composable_kernel_mfma")
    # bench.bench(benchmark_func(ck_gemm_dl), "composable_kernel_simt")
    # bench.bench(benchmark_func(mfma_gemmv1), "mfma_v1")
    # bench.bench(benchmark_func(lambda a, b: mfma_gemmv1(a, b, version=1)), "mfma_v2")
    # bench.bench(benchmark_func(wmma_gemm), "wmma_v1")
    # bench.bench(benchmark_func(mfma_gemmv2), "wmma_mfma")
    # bench.bench(benchmark_func(lambda a, b: mfma_gemmv2(a, b, ver=1, pack_len=4)), "wmma_mfma_v1_pack4")
    # bench.bench(benchmark_func(lambda a, b: mfma_gemmv2(a, b, ver=1, pack_len=2)), "wmma_mfma_v1_pack2")
    bench.bench(benchmark_func(lambda a, b: mfma_gemmv3(a, b, ver=4)), "mfma_v3_ver4")
    # bench.bench(benchmark_func(lambda a, b: mfma_gemmv3(a, b, ver=1)), "mfma_v3_ver1")
    # bench.bench(benchmark_func(lambda a, b: mfma_gemmv3(a, b, ver=2)), "mfma_v3_ver2")
    # bench.bench(benchmark_func(lambda a, b: mfma_gemmv3(a, b, ver=3)), "mfma_v3_ver3")
    # bench.bench(benchmark_func(lambda a, b: mfma_gemmv3(a, b, ver=4)), "mfma_v3_ver4")
    bench.bench(benchmark_func(lambda a, b: mfma_gemmv3(a, b, ver=5)), "mfma_v3_ver5")
    # bench.bench(benchmark_func(lambda a, b: mfma_gemmv3_5(a, b, ver=0)), "mfma_v3_5_ver0")
    # bench.bench(benchmark_func(lambda a, b: mfma_gemmv3_5(a, b, ver=1)), "mfma_v3_5_ver1")
    # bench.bench(benchmark_func(lambda a, b: mfma_gemmv3(a, b, ver=6)), "mfma_v3_ver6")
    
    
    
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


#           blas   mfma_v2   wmma_v1  mfma_v3_ver0  mfma_v3_ver2  mfma_v3_ver4  \
# 256   0.017280  0.036000  0.017920      0.019840      0.019840      0.022240   
# 512   0.026720  0.066240  0.035040      0.031520      0.031520      0.044480   
# 768   0.044000  0.101440  0.061280      0.056480      0.056480      0.095520   
# 1024  0.083520  0.182721  0.124001      0.100640      0.101440      0.186721   
# 1280  0.154880  0.256162  0.168961      0.169120      0.157761      0.246401   
# 1536  0.255201  0.398402  0.279842      0.287041      0.287041      0.477123   
# 1792  0.389442  0.512163  0.420642      0.393842      0.394802      0.585123   
# 2048  0.522802  0.794724  0.646644      0.596163      0.634163      1.024325   

#       mfma_v3_ver5  mfma_v3_ver6  
# 256       0.017760      0.017760  
# 512       0.030080      0.029600  
# 768       0.055200      0.053921  
# 1024      0.106880      0.106560  
# 1280      0.179041      0.176961  
# 1536      0.278722      0.279361  
# 1792      0.430882      0.432322  
# 2048      0.614723      0.615043 


#           blas  composable_kernel_mfma  mfma_v3_ver0  mfma_v3_ver5
# 2048  0.520000                0.546800      0.595042      0.613923
# 2560  1.129285                1.029605      1.094726      1.178246
# 3072  1.962571                1.802570      1.919212      2.034412
# 3584  3.153939                2.834577      3.014416      3.180256
# 4096  4.363143                4.133061      4.423062      4.632824


# Nov 28

#           blas  composable_kernel_mfma  mfma_v3_ver0  mfma_v3_ver4
# 256   0.017280                0.030880      0.025920      0.023200
# 512   0.026720                0.049760      0.039840      0.034240
# 768   0.044000                0.070080      0.061440      0.063040
# 1024  0.083200                0.124080      0.109760      0.103680
# 1280  0.154080                0.160481      0.159361      0.161120
# 1536  0.254241                0.259680      0.265760      0.260640
# 1792  0.389441                0.383361      0.393921      0.382241
# 2048  0.525281                0.552322      0.579522      0.542562

#           blas  composable_kernel_mfma  mfma_v3_ver0  mfma_v3_ver4
# 2048  0.519521                0.548001      0.578401      0.537761
# 2560  1.125923                1.020162      1.090562      1.068802
# 3072  1.952324                1.781764      1.893283      1.871764
# 3584  3.140646                2.810246      2.910006      2.904486
# 4096  4.357929                4.102889      4.345609      4.243129