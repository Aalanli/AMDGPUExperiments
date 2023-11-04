# %%
import torch
from kernels import KernelHandler, KernelConfig
from kernels.gemm import rocgemm, simt_gemm
from kernels.gemmv2 import simt_gemmv2
import hidet
from kernels.utils import Bench

if __name__ == '__main__':
    def bench_rocgemm1(i, **kwargs):
        m, k, n = i
        a = torch.randn([m, k], device='cuda')
        b = torch.randn([k, n], device='cuda')
        return lambda: rocgemm(a, b, version=1)
    
    def bench_simtgemm(i, **kwargs):
        m, k, n = i
        a = torch.randn([m, k], device='cuda')
        b = torch.randn([k, n], device='cuda')
        return lambda: simt_gemm(a, b)
    
    def bench_simtgemmv2(i, **kwargs):
        m, k, n = i
        a = torch.randn([m, k], device='cuda')
        b = torch.randn([k, n], device='cuda')
        return lambda: simt_gemmv2(a, b)

    def bench_rocgemm2(i, **kwargs):
        m, k, n = i
        a = torch.randn([m, k], device='cuda')
        b = torch.randn([k, n], device='cuda')
        return lambda: rocgemm(a, b, version=2)

    def bench_blas(i, **kwargs):
        m, k, n = i
        a = torch.randn([m, k], device='cuda')
        b = torch.randn([k, n], device='cuda')
        return lambda: a @ b
    
    def bench_hidet(i, **kwargs):
        m, n, k = i
        a = hidet.from_torch(torch.randn([1, m, k], device='cuda'))
        b = hidet.from_torch(torch.randn([1, k, n], device='cuda'))
        a1 = hidet.symbol([1, 'm', 'k'], dtype='float32', device='cuda')
        b1 = hidet.symbol([1, 'k', 'n'], dtype='float32', device='cuda')
        from hidet.graph.ops.matmul.batch_matmul import batch_matmul
        c = batch_matmul(a1, b1)
        g = hidet.trace_from(c, [a1, b1])
        g = hidet.graph.optimize(g)
        g = g.build(space=2)

        return lambda: g(a, b)
    
    square = [(m, m, m) for m in range(256, 2048 + 1024, 128)]
    mem_bound = [(1, m, m) for m in [32, 64, 128, 256, 512, 768, 1024]]
    bench = Bench(
        x_vals=square,
        x_name='(m, k, n)',
    )
    # bench.bench(bench_rocgemm1)
    # bench.bench(bench_rocgemm2, 'naive')
    bench.bench(bench_blas, 'blas')
    bench.bench(bench_simtgemm, 'simt')
    bench.bench(bench_simtgemmv2, 'simtv2')
    bench.bench(bench_hidet, 'hidet')
    data = bench.run()
    data.show_plot()
