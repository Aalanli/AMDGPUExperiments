# %%
import torch
from kernels import KernelHandler, KernelConfig

kernel = KernelHandler(
    source_file='src/rocblas_gemm.cpp',
    compile_configs=[
        KernelConfig({'BLOCKSIZE_M': 64, 'BLOCKSIZE_N': 64, 'BLOCKSIZE_K': 16, 'WARPSZ_M': 32, 'WARPSZ_N': 8, 'READ_A_DIM': 64, 'READ_B_DIM': 64, 'TYPE': 'float'})
    ],
    keys=['m', 'n', 'k'],
    platform='nvidia'
)

def rocgemm(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    assert a.shape[1] == b.shape[0]
    assert len(a.shape) == len(b.shape) == 2
    m, k = a.shape
    n = b.shape[1]
    c = torch.empty((m, n), device=a.device, dtype=a.dtype)
    kernel(a, b, c, m=m, n=n, k=k, version=1)
    return c

# %%
a = torch.randn([512, 512], device='cuda')
b = torch.randn([512, 512], device='cuda')
c = rocgemm(a, b)
