# %%
from typing import Optional
import torch
from kernels import KernelHandler, KernelConfig

# max smem bytes = 65536
def generate_configs():
    for block_n in [64, 128, 256]:
        for block_m in [64, 128, 256]:
            for block_k in [32, 64]:
                for inner_k in [8]:
                    for vec_load in [1, 2, 4, 8]:
                        if block_k % inner_k != 0: continue
                        for warps_m, warps_n in [(2, 1), (2, 2), (4, 1), (8, 1), (4, 2)]:
                            warps = warps_m * warps_n
                            if block_m % (32 * warps_m) != 0: continue
                            if block_n % (32 * warps_n) != 0: continue

                            if warps * 64 < block_k: continue
                            if warps * 64 < block_n: continue
                            yield KernelConfig(
                                {'_BLOCK_N': block_n, 
                                '_BLOCK_M': block_m, 
                                '_BLOCK_K': block_k, 
                                '_WARP_M': warps_m,
                                '_WARP_N': warps_n,
                                '_VecLoad': vec_load,
                                '_InnerK': inner_k,
                                })

configs = [
    KernelConfig({'_BLOCK_N': 64, '_BLOCK_M': 128, '_BLOCK_K': 32, '_WARP_M': 2, '_WARP_N': 1, '_VecLoad': 8, '_InnerK': 8, '_SMEM_INNER_SWIZZLE': 1}),
    # KernelConfig({'_BLOCK_N': 128, '_BLOCK_M': 128, '_BLOCK_K': 16, '_Warps': 4, '_VecLoad': 4, '_InnerK': 8, '_SMEM_INNER_SWIZZLE': 1}),
]

kernel = KernelHandler(
    source_file='src/mma_gemm/mfma_gemmf16v3.cpp', 
    compile_configs=list(generate_configs()),
    keys=['m', 'k', 'n', 'ver'],
    platform='amd',
    disable_benchmark=False,
    ignore_compile_errors=True,
    parallel_compile=True,
    arch='gfx90a',
)

def mfma_gemmv3f16(a: torch.Tensor, b: torch.Tensor, ver: int = 0, so_name: Optional[str] = None) -> torch.Tensor:
    assert a.shape[1] == b.shape[0]
    assert len(a.shape) == len(b.shape) == 2
    m, k = a.shape
    n = b.shape[1]
    c = torch.empty((m, n), device=a.device, dtype=a.dtype)
    if so_name is None:
        kernel(a, b, c, m=m, k=k, n=n, ver=ver)
    else:
        kernel.call_so(so_name, a, b, c, m=m, k=k, n=n, ver=ver)
    return c

def show_err(x):
    from matplotlib import pyplot as plt
    plt.imshow(x.float().cpu().numpy())
    plt.show()

if __name__ == '__main__':
    # peak: 0.19424
    d = 2048
    # a = torch.arange(0, d, device='cuda')[None, :] + torch.arange(0, d, device='cuda')[:, None] * d
    # a = a.to(torch.half)
    # b = torch.eye(d, device='cuda').to(torch.half)
    a = torch.randn([d, d], device='cuda', dtype=torch.half)
    b = torch.randn([d, d], device='cuda', dtype=torch.half)
    c1 = a @ b
    c = mfma_gemmv3f16(a, b, ver=0)
    err = (c1 - c).abs()
    print(c)
    print(err)
    print(err.max())
    from kernels.build_kernel import do_bench
    print(do_bench(lambda: mfma_gemmv3f16(a, b, ver=0)))
    print(do_bench(lambda: mfma_gemmv3f16(a, b, ver=1)))
    print(do_bench(lambda: mfma_gemmv3f16(a, b, ver=2)))
    print(do_bench(lambda: mfma_gemmv3f16(a, b, ver=3)))
    