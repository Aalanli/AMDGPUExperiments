# %%
import torch
from kernels import KernelHandler, KernelConfig
from kernels.gemm import rocgemm, simt_gemm
from kernels.gemmv2 import simt_gemmv2
from kernels.gemmv3 import simt_gemmv3
from kernels.gemmv4 import simt_gemmv4
from kernels.gemmv5 import simt_gemmv5
from kernels.gemm_hidet import hidet_simt
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
    
    def bench_simtgemmv3(i, **kwargs):
        m, k, n = i
        a = torch.randn([m, k], device='cuda')
        b = torch.randn([k, n], device='cuda')
        return lambda: simt_gemmv3(a, b)
    
    def bench_simtgemmv4(i, **kwargs):
        m, k, n = i
        a = torch.randn([m, k], device='cuda')
        b = torch.randn([k, n], device='cuda')
        return lambda: simt_gemmv4(a, b)

    def bench_simtgemmv5(i, **kwargs):
        m, k, n = i
        a = torch.randn([m, k], device='cuda')
        b = torch.randn([k, n], device='cuda')
        return lambda: simt_gemmv5(a, b)
    
    def bench_simt_hidet(i, **kwargs):
        m, k, n = i
        a = torch.randn([m, k], device='cuda')
        b = torch.randn([k, n], device='cuda')
        return lambda: hidet_simt(a, b)
    

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
    
    square = [(m, m, m) for m in range(256, 2048, 256)]
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
    bench.bench(bench_simtgemmv3, 'simtv3')
    bench.bench(bench_simtgemmv4, 'simtv4')
    bench.bench(bench_simtgemmv5, 'simtv5')
    bench.bench(bench_simt_hidet, 'simt_hidet')
    bench.bench(bench_hidet, 'hidet')
    data = bench.run()
    data.show_plot()
    data.print_data()

# RTX 3090
#                         blas      simt    simtv2    simtv3    simtv4  \
# (256, 256, 256)     0.011264  0.014336  0.013312  0.016512  0.020704   
# (512, 512, 512)     0.023552  0.030720  0.032768  0.033792  0.038912   
# (768, 768, 768)     0.052224  0.067584  0.081920  0.075776  0.074752   
# (1024, 1024, 1024)  0.103424  0.162816  0.182272  0.162816  0.157696   
# (1280, 1280, 1280)  0.198656  0.259072  0.330752  0.280576  0.277504   
# (1536, 1536, 1536)  0.295936  0.493440  0.601088  0.530432  0.499760   
# (1792, 1792, 1792)  0.525824  0.755712  0.972800  0.808960  0.771072   

#                       simtv5  simt_hidet     hidet  
# (256, 256, 256)     0.021504    0.014336  0.019168  
# (512, 512, 512)     0.037648    0.031744  0.032768  
# (768, 768, 768)     0.070656    0.061440  0.063488  
# (1024, 1024, 1024)  0.161792    0.134944  0.126976  
# (1280, 1280, 1280)  0.270336    0.211968  0.225280  
# (1536, 1536, 1536)  0.507904    0.379904  0.376096  
# (1792, 1792, 1792)  0.766976    0.607120  0.594944  

