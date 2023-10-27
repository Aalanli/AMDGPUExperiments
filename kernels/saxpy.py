# %%
import math
import torch
from kernels import KernelHandler, KernelConfig

kernel = KernelHandler(
    source_file='src/saxpy.cu', 
    compile_configs=[
        KernelConfig({'BLOCKSIZE': 512, 'REPEATS': 4}),
        KernelConfig({'BLOCKSIZE': 256, 'REPEATS': 8}),
        KernelConfig({'BLOCKSIZE': 1024, 'REPEATS': 2}),
        KernelConfig({'BLOCKSIZE': 1024, 'REPEATS': 4}),
    ], 
    keys=['n', 'd'], 
    platform='nvidia'
)

def saxpy(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    assert a.shape == b.shape
    d = a.shape[-1]
    n = math.prod(a.shape[:-1])
    c = torch.empty_like(a)
    kernel(a, b, c, n=n, d=d)
    return c

if __name__ == '__main__':
    a = torch.randn(1000, device='cuda')
    b = torch.randn(1000, device='cuda')
    c = saxpy(a, b)
    c1 = a + b
    print(torch.allclose(c, c1))

    a = torch.randn(2, 1000, device='cuda')
    b = torch.randn(2, 1000, device='cuda')
    c = torch.zeros_like(a)
    c = saxpy(a, b)
    c1 = a + b
    print(torch.allclose(c, c1))

