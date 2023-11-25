# %%
import torch
from kernels import KernelHandler, KernelConfig

def generate_configs():
    for block_n in [16, 32, 64, 128]:
        for block_m in [16, 32, 64, 128]:
            for block_k in [4, 8, 16, 32]:
                for inner_k in [4, 8, 16]:
                    for vec_load in [1, 2, 4]:
                        if block_k % inner_k != 0: continue
                        for warps in [1, 2, 4]:
                            warps_m = min(block_m // 16, warps)
                            warps_n = min(block_n // 16, warps // warps_m)
                            if warps_m * warps_n != warps: continue
                            yield KernelConfig(
                                {'_BLOCK_N': block_n, 
                                 '_BLOCK_M': block_m, 
                                 '_BLOCK_K': block_k, 
                                 '_Warps': warps,
                                 '_VecLoad': vec_load,
                                 '_InnerK': inner_k
                                })

configs = [
    # KernelConfig({'_BLOCK_N': 32, '_BLOCK_M': 64, '_BLOCK_K': 4, '_Warps': 4, '_VecLoad': 4, '_InnerK': 4}),
    KernelConfig({'_BLOCK_N': 32, '_BLOCK_M': 16, '_BLOCK_K': 16, '_Warps': 1, '_VecLoad': 4, '_InnerK': 8}),
    # KernelConfig({'_BLOCK_N': 32, '_BLOCK_M': 16, '_BLOCK_K': 8, '_Warps': 1}),
    # KernelConfig({'_BLOCK_N': 32, '_BLOCK_M': 32, '_BLOCK_K': 8, '_Warps': 1}),
    # KernelConfig({'_BLOCK_N': 32, '_BLOCK_M': 16, '_BLOCK_K': 8, '_Warps': 1}),
    # KernelConfig({'_BLOCK_N': 32, '_BLOCK_M': 16, '_BLOCK_K': 8, '_Warps': 1}),
]

kernel = KernelHandler(
    source_file='src/mma_gemm/mfma_gemmv3.cpp', 
    compile_configs=list(generate_configs()),
    keys=['m', 'k', 'n', 'ver'],
    platform='amd',
    disable_benchmark=False,
    ignore_compile_errors=True,
    parallel_compile=True
)

def mfma_gemmv3(a: torch.Tensor, b: torch.Tensor, ver: int = 0) -> torch.Tensor:
    assert a.shape[1] == b.shape[0]
    assert len(a.shape) == len(b.shape) == 2
    m, k = a.shape
    n = b.shape[1]
    c = torch.empty((m, n), device=a.device, dtype=a.dtype)
    kernel(a, b, c, m=m, k=k, n=n, ver=ver)
    return c


if __name__ == '__main__':
    a = torch.arange(0, 32, device='cuda')[None, :] + torch.arange(0, 32, device='cuda')[:, None] * 32
    a = a.float()
    a = torch.randn([512, 512], device='cuda')
    b = torch.randn([512, 512], device='cuda')
    # b = torch.eye(32, device='cuda')
    c1 = a @ b
    c = mfma_gemmv3(a, b, ver=1)
    err = (c1 - c).abs()
    print(c)
    print(err)
    print(err.max())

