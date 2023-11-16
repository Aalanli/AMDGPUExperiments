# %%
import torch
from kernels import KernelHandler, KernelConfig

configs = [
    KernelConfig({'BLOCK_M': 16, 'BLOCK_N': 16, 'BLOCK_K': 16, 'Warp_M': 2, 'Warp_N': 2})
]

kernel = KernelHandler(
    source_file='src/mfma_gemm.cpp', 
    compile_configs=configs,
    keys=['m', 'k', 'n'],
    platform='amd',
    disable_benchmark=True,
    ignore_compile_errors=False,
    parallel_compile=False
)

def mfma_gemmv1(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    assert a.shape[1] == b.shape[0]
    assert len(a.shape) == len(b.shape) == 2
    m, k = a.shape
    n = b.shape[1]
    c = torch.zeros((m, n), device=a.device, dtype=a.dtype) - 1
    kernel(a, b, c, m=m, k=k, n=n)
    return c


if __name__ == '__main__':
    d = 64
    a = torch.arange(0, 64, device='cuda')[None, :] + torch.arange(0, 64, device='cuda')[:, None] * 64
    a = a.float()
    # a = torch.randn([64, 64], device='cuda')
    b = torch.ones([64, 64], device='cuda')
    c1 = a @ b
    c = mfma_gemmv1(a, b)
    err = (c1 - c).abs()
    print(c)
    print(err)
    print(err.max())

