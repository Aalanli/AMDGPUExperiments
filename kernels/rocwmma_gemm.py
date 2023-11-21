# %%
import torch
from kernels import KernelHandler, KernelConfig

def generate_configs():
    for mma_m, mma_n in [(16, 16), (32, 32)]:
        for mma_k in [4, 8, 16, 32, 64]:
            for rep_m, rep_n in [(1, 1), (1, 2), (2, 1), (2, 2)]:
                for warp_m, warp_n in [(1, 1), (1, 2), (2, 1), (2, 2)]:
                    for nstages in [1, 2]:
                        for unroll_k in [0, 1]:
                            yield KernelConfig({'MMA_M': mma_m, 'MMA_N': mma_n, 'MMA_K': mma_k, 
                                                'REP_M': rep_m, 'REP_N': rep_n, 'WARP_M': warp_m,
                                                'WARP_N': warp_n, 'NSTAGES': nstages, 'UNROLL_LASTK': unroll_k})
configs = [
    KernelConfig({'MMA_M': 16, 'MMA_N': 16, 'MMA_K': 8, 'REP_M': 1, 
                  'REP_N': 1, 'WARP_M': 1, 'WARP_N': 1, 'NSTAGES': 2, 'UNROLL_LASTK': 1})
]

kernel = KernelHandler(
    source_file='src/mma_gemm/rocwmma_gemm.cpp',
    compile_configs=list(generate_configs()),
    keys=['m', 'k', 'n'],
    platform='amd',
    disable_benchmark=False,
    ignore_compile_errors=True,
    parallel_compile=True
)

def wmma_gemm(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    assert a.shape[1] == b.shape[0]
    assert len(a.shape) == len(b.shape) == 2
    m, k = a.shape
    n = b.shape[1]
    c = torch.empty((m, n), device=a.device, dtype=a.dtype)
    kernel(a, b, c, m=m, k=k, n=n)
    return c


if __name__ == '__main__':
    a = torch.arange(0, 64, device='cuda')[None, :] + torch.arange(0, 64, device='cuda')[:, None] * 64
    a = a.float()
    a = torch.randn([512, 512], device='cuda')
    b = torch.randn([512, 512], device='cuda')
    # b = torch.eye(64, device='cuda')
    c1 = a @ b
    c = wmma_gemm(a, b)
    err = (c1 - c).abs()
    print(c)
    print(err)
    print(err.max())



# %%
