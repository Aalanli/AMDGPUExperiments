# %%
import torch
# from kernels.gemm import rocgemm
from kernels.gemm_mfmaf16_v1 import mfma_gemmv1f16
# from kernels.rocwmma_gemm import wmma_gemm
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
    
    square = [m for m in range(256, 2049, 256)]
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
    bench.bench(benchmark_func(lambda a, b: mfma_gemmv1f16(a, b, ver=0)), "mfmaf16v1_ver0")
    bench.bench(benchmark_func(lambda a, b: mfma_gemmv1f16(a, b, ver=1)), "mfmaf16v1_ver0")

    data = bench.run()
    data.show_plot()
    data.print_data()

