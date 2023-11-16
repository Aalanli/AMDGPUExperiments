# %%
import torch
from kernels import KernelHandler, KernelConfig

def generate_configs():
    for rep_m, rep_n in [(1, 1), (1, 2), (2, 1), (2, 2), (2, 4), (4, 2)]:
        for rep_k in [2, 4, 8]:
            for warp_m, warp_n in [(1, 1), (1, 2), (2, 1), (2, 2)]:
                nthreads = warp_m * warp_n * 64
                block_m = rep_m * warp_m * 16
                block_n = rep_n * warp_n * 16
                block_k = rep_k * 4
                if nthreads % block_k != 0 or block_k > nthreads: continue
                if (block_m + block_n) * block_k * 4 > 65536: continue
                yield KernelConfig({'REP_M': rep_m, 'REP_N': rep_n, 'REP_K': rep_k, 'Warp_M': warp_m, 'Warp_N': warp_n})



configs = [
    KernelConfig({'REP_M': 1, 'REP_N': 1, 'REP_K': 4, 'Warp_M': 1, 'Warp_N': 1}),
    KernelConfig({'REP_M': 1, 'REP_N': 1, 'REP_K': 4, 'Warp_M': 1, 'Warp_N': 2}),
    KernelConfig({'REP_M': 1, 'REP_N': 1, 'REP_K': 4, 'Warp_M': 2, 'Warp_N': 2}),
    KernelConfig({'REP_M': 1, 'REP_N': 1, 'REP_K': 4, 'Warp_M': 2, 'Warp_N': 2}),
    KernelConfig({'REP_M': 1, 'REP_N': 1, 'REP_K': 4, 'Warp_M': 2, 'Warp_N': 2}),
]

kernel = KernelHandler(
    source_file='src/mfma_gemm.cpp', 
    compile_configs=list(generate_configs()),
    keys=['m', 'k', 'n'],
    platform='amd',
    disable_benchmark=False,
    ignore_compile_errors=True,
    parallel_compile=True
)

def mfma_gemmv1(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    assert a.shape[1] == b.shape[0]
    assert len(a.shape) == len(b.shape) == 2
    m, k = a.shape
    n = b.shape[1]
    c = torch.empty((m, n), device=a.device, dtype=a.dtype)
    kernel(a, b, c, m=m, k=k, n=n)
    return c


if __name__ == '__main__':
    d = 64
    a = torch.arange(0, 64, device='cuda')[None, :] + torch.arange(0, 64, device='cuda')[:, None] * 64
    a = a.float()
    a = torch.randn([512, 512], device='cuda')
    b = torch.randn([512, 512], device='cuda')
    # b = torch.eye(64, device='cuda')
    c1 = a @ b
    c = mfma_gemmv1(a, b)
    err = (c1 - c).abs()
    print(c)
    print(err)
    print(err.max())

