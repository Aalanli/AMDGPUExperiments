# %%
import torch
from kernels import KernelHandler, KernelConfig
from kernels.gemm_hidet import hidet_simt
from kernels.utils import Bench

if __name__ == '__main__':    
    def bench_simt_hidet(i, **kwargs):
        m, k, n = i
        a = torch.randn([m, k], device='cuda')
        b = torch.randn([k, n], device='cuda')
        return lambda: hidet_simt(a, b)
    
    def bench_simt_hidetv2(i, **kwargs):
        m, k, n = i
        a = torch.randn([m, k], device='cuda')
        b = torch.randn([k, n], device='cuda')
        return lambda: hidet_simt(a, b, version=1)
    
    def bench_simt_hidetv3(i, **kwargs):
        m, k, n = i
        a = torch.randn([m, k], device='cuda')
        b = torch.randn([k, n], device='cuda')
        return lambda: hidet_simt(a, b, version=2)
    
    def bench_blas(i, **kwargs):
        m, k, n = i
        a = torch.randn([m, k], device='cuda')
        b = torch.randn([k, n], device='cuda')
        return lambda: a @ b
    
    
    square = [(m, m, m) for m in range(256, 2049, 256)]
    mem_bound = [(1, m, m) for m in [32, 64, 128, 256, 512, 768, 1024]]
    bench = Bench(
        x_vals=square,
        x_name='(m, k, n)',
    )
    # bench.bench(bench_rocgemm1)
    # bench.bench(bench_rocgemm2, 'naive')
    bench.bench(bench_blas, 'blas')
    bench.bench(bench_simt_hidet, 'simt_hidet')
    bench.bench(bench_simt_hidetv2, 'simt_hidetv2')
    bench.bench(bench_simt_hidetv3, 'simt_hidetv3')
    data = bench.run()
    data.show_plot()
    data.print_data()

