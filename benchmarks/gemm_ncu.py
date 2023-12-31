# %%
from kernels.gemm import simt_gemm, rocgemm
from kernels.gemmv2 import simt_gemmv2
from kernels.gemmv3 import simt_gemmv3
from kernels.gemmv4 import simt_gemmv4
from kernels.gemmv5 import simt_gemmv5
from kernels.gemm_hidet import hidet_simt
from kernels.gemmv6 import simt_gemmv6
import torch

a = torch.randn([1024, 1024], device='cuda')
b = torch.randn([1024, 1024], device='cuda')

simt_gemm(a, b)
simt_gemmv2(a, b)
simt_gemmv3(a, b)
simt_gemmv4(a, b)
simt_gemmv5(a, b)
hidet_simt(a, b, version=0)
hidet_simt(a, b, version=1)
hidet_simt(a, b, version=2)
hidet_simt(a, b, version=3)
simt_gemmv6(a, b)
rocgemm(a, b, version=2)

import hidet
hidet.option.cache_dir('./outs/simt_matmul')
hidet.option.debug_cache_tuning()
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

bench_hidet((1024, 1024, 1024))()
