# %%
import torch
# from kernels.gemm import rocgemm
from kernels.gemm_mfmaf16_v1 import mfma_gemmv1f16
from kernels.gemm_mfmaf16v2 import mfma_gemmv2f16
from kernels.gemm_mfmaf16v3 import mfma_gemmv3f16
from composable_kernel_gemmf16 import ck_gemmf16
# from kernels.rocwmma_gemm import wmma_gemm
from kernels.utils import Bench
from triton.ops.matmul import matmul

if __name__ == '__main__':
    def benchmark_func(f):
        def bench_fn(i, **kwargs):
            m, k, n = i, i, i
            a = torch.randn([m, k], device='cuda', dtype=torch.half)
            b = torch.randn([k, n], device='cuda', dtype=torch.half)
            return lambda: f(a, b)
        return bench_fn
    
    square = [m for m in range(2048, 4097, 512)]
    bench = Bench(
        x_vals=square,
        x_name='(m, k, n)',
    )
    bench.bench(benchmark_func(lambda a, b: a @ b), 'blas')
    # bench.bench(benchmark_func(matmul), 'triton')
    # bench.bench(benchmark_func(lambda a, b: mfma_gemmv1f16(a, b, ver=0)), "mfmaf16v1_ver0")
    # bench.bench(benchmark_func(lambda a, b: mfma_gemmv1f16(a, b, ver=1)), "mfmaf16v1_ver1")

    # bench.bench(benchmark_func(lambda a, b: mfma_gemmv2f16(a, b, ver=0)), "mfmaf16v2_ver0")
    # bench.bench(benchmark_func(lambda a, b: mfma_gemmv2f16(a, b, ver=1)), "mfmaf16v2_ver1")

    # bench.bench(benchmark_func(lambda a, b: mfma_gemmv2f16(a, b, ver=2)), "mfmaf16v2_ver2")
    bench.bench(benchmark_func(lambda a, b: mfma_gemmv2f16(a, b, ver=3)), "mfmaf16v2_ver3")
    bench.bench(benchmark_func(lambda a, b: mfma_gemmv3f16(a, b, ver=0)), "mfmaf16v2_ver0")
    bench.bench(benchmark_func(lambda a, b: mfma_gemmv3f16(a, b, ver=1)), "mfmaf16v2_ver1")
    bench.bench(benchmark_func(lambda a, b: mfma_gemmv3f16(a, b, ver=2)), "mfmaf16v2_ver2")
    bench.bench(benchmark_func(lambda a, b: mfma_gemmv3f16(a, b, ver=3)), "mfmaf16v2_ver3")
    bench.bench(benchmark_func(ck_gemmf16), 'composable_kernel')

    data = bench.run()
    data.show_plot()
    data.print_data()

