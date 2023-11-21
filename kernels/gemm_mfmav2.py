# %%
import torch
from kernels import KernelHandler, KernelConfig

def generate_configs():
    for block_n in [16, 32, 64, 128]:
        for block_m in [16, 32, 64, 128]:
            for block_k in [4, 8, 16, 32]:
                for warps in [1, 2, 4]:
                    if block_n * block_m // 16 < warps: continue
                    yield KernelConfig(
                        {'_BLOCK_N': block_n, '_BLOCK_M': block_m, '_BLOCK_K': block_k, '_Warps': warps})

configs = [
    KernelConfig({'_BLOCK_N': 32, '_BLOCK_M': 32, '_BLOCK_K': 16, '_Warps': 2}),
    # KernelConfig({'_BLOCK_N': 32, '_BLOCK_M': 16, '_BLOCK_K': 8, '_Warps': 1}),
    # KernelConfig({'_BLOCK_N': 32, '_BLOCK_M': 32, '_BLOCK_K': 8, '_Warps': 1}),
    # KernelConfig({'_BLOCK_N': 32, '_BLOCK_M': 16, '_BLOCK_K': 8, '_Warps': 1}),
    # KernelConfig({'_BLOCK_N': 32, '_BLOCK_M': 16, '_BLOCK_K': 8, '_Warps': 1}),
]

kernel = KernelHandler(
    source_file='src/mma_gemm/mfma_gemmv2.cpp', 
    compile_configs=list(generate_configs()),
    keys=['m', 'k', 'n'],
    platform='amd',
    disable_benchmark=False,
    ignore_compile_errors=True,
    parallel_compile=True
)

def mfma_gemmv2(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    assert a.shape[1] == b.shape[0]
    assert len(a.shape) == len(b.shape) == 2
    m, k = a.shape
    n = b.shape[1]
    c = torch.zeros((m, n), device=a.device, dtype=a.dtype) - 1
    kernel(a, b, c, m=m, k=k, n=n)
    return c


if __name__ == '__main__':
    a = torch.arange(0, 32, device='cuda')[None, :] + torch.arange(0, 32, device='cuda')[:, None] * 32
    a = a.float()
    a = torch.randn([1024, 1024], device='cuda')
    b = torch.randn([1024, 1024], device='cuda')
    # b = torch.eye(32, device='cuda')
    c1 = a @ b
    c = mfma_gemmv2(a, b)
    err = (c1 - c).abs()
    print(c)
    print(err)
    print(err.max())

