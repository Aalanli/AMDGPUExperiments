# %%
import torch
from kernels import KernelHandler, KernelConfig
from kernels import Bench

kernel_rocblas = KernelHandler(
    source_file='src/rocblas_gemm.cpp',
    compile_configs=[
        KernelConfig({'BLOCKSIZE_M': 64, 'BLOCKSIZE_N': 64, 'BLOCKSIZE_K': 16, 'WARPSZ_M': 32, 'WARPSZ_N': 8, 'READ_A_DIM': 16, 'READ_B_DIM': 64, 'TYPE': 'float'}),
        KernelConfig({'BLOCKSIZE_M': 64, 'BLOCKSIZE_N': 64, 'BLOCKSIZE_K': 16, 'WARPSZ_M': 64, 'WARPSZ_N': 8, 'READ_A_DIM': 16, 'READ_B_DIM': 64, 'TYPE': 'float'}),
        KernelConfig({'BLOCKSIZE_M': 64, 'BLOCKSIZE_N': 64, 'BLOCKSIZE_K': 32, 'WARPSZ_M': 64, 'WARPSZ_N': 8, 'READ_A_DIM': 32, 'READ_B_DIM': 64, 'TYPE': 'float'}),
        KernelConfig({'BLOCKSIZE_M': 64, 'BLOCKSIZE_N': 64, 'BLOCKSIZE_K': 32, 'WARPSZ_M': 32, 'WARPSZ_N': 16, 'READ_A_DIM': 32, 'READ_B_DIM': 64, 'TYPE': 'float'}),
        KernelConfig({'BLOCKSIZE_M': 64, 'BLOCKSIZE_N': 64, 'BLOCKSIZE_K': 32, 'WARPSZ_M': 16, 'WARPSZ_N': 16, 'READ_A_DIM': 32, 'READ_B_DIM': 64, 'TYPE': 'float'}),
        KernelConfig({'BLOCKSIZE_M': 128, 'BLOCKSIZE_N': 64, 'BLOCKSIZE_K': 16, 'WARPSZ_M': 32, 'WARPSZ_N': 16, 'READ_A_DIM': 16, 'READ_B_DIM': 64, 'TYPE': 'float'}),
        KernelConfig({'BLOCKSIZE_M': 64, 'BLOCKSIZE_N': 128, 'BLOCKSIZE_K': 16, 'WARPSZ_M': 16, 'WARPSZ_N': 32, 'READ_A_DIM': 16, 'READ_B_DIM': 64, 'TYPE': 'float'}),
        KernelConfig({'BLOCKSIZE_M': 128, 'BLOCKSIZE_N': 64, 'BLOCKSIZE_K': 32, 'WARPSZ_M': 32, 'WARPSZ_N': 16, 'READ_A_DIM': 32, 'READ_B_DIM': 64, 'TYPE': 'float'}),
        KernelConfig({'BLOCKSIZE_M': 64, 'BLOCKSIZE_N': 128, 'BLOCKSIZE_K': 32, 'WARPSZ_M': 16, 'WARPSZ_N': 32, 'READ_A_DIM': 32, 'READ_B_DIM': 64, 'TYPE': 'float'}),
        KernelConfig({'BLOCKSIZE_M': 64, 'BLOCKSIZE_N': 64, 'BLOCKSIZE_K': 64, 'WARPSZ_M': 32, 'WARPSZ_N': 32, 'READ_A_DIM': 64, 'READ_B_DIM': 64, 'TYPE': 'float'}),
    ],
    keys=['m', 'n', 'k', 'version'],
    platform='nvidia',
    disable_benchmark=False
)

def rocgemm(a: torch.Tensor, b: torch.Tensor, version: int = 1) -> torch.Tensor:
    assert a.shape[1] == b.shape[0]
    assert len(a.shape) == len(b.shape) == 2
    m, k = a.shape
    n = b.shape[1]
    c = torch.empty((m, n), device=a.device, dtype=a.dtype)
    kernel_rocblas(a, b, c, m=m, k=k, n=n, version=version)
    return c

def gen_configs():
    for (bwm, bwn) in [(1, 1), (2, 2), (1, 2), (2, 1), (2, 4), (4, 2)]:
        for (wm, wn, wk) in [(4, 8, 1), (2, 16, 1), (16, 2, 1), (8, 4, 1)]:
            for (tm, tn, tk) in [(4, 4, 2), (4, 4, 1), (2, 2, 2), (2, 2, 1)]:
                for block_k in [16, 32, 64]:
                    block_m = bwm * wm * tm
                    block_n = bwn * wn * tn

                    for (block_m, block_n) in [(block_m, block_n), (block_m * 2, block_n), (block_m, block_n * 2), (block_m * 2, block_n * 2)]:
                        smem = max(block_m * block_k + block_k * block_n, block_m * block_n) * 4
                        regs = (tk * (tm + tn) + tm * tn) * 4 + 32
                        if regs > 255:
                            continue
                        if smem > 0xc000:
                            continue
                        yield KernelConfig({
                            'BlockM': block_m,
                            'BlockK': block_k,
                            'BlockN': block_n,
                            'WarpM': bwm,
                            'WarpN': bwn,
                            'ThreadM': wm,
                            'ThreadK': wk,
                            'ThreadN': wn,
                            'TM': tm,
                            'TN': tn,
                            'TK': tk,
                            'TYPE': 'float'
                        })

hand_picked_configs = [
    KernelConfig({'BlockM': 8, 'BlockK': 32, 'BlockN': 32, 'TM': 2, 'TN': 2, 'TK': 1, 'ThreadM': 2, 'ThreadK': 1, 'ThreadN': 16, 'WarpM': 1, 'WarpN': 1, 'TYPE': 'float'}),
    KernelConfig({'BlockM': 16, 'BlockK': 32, 'BlockN': 64, 'TM': 4, 'TN': 4, 'TK': 1, 'ThreadM': 2, 'ThreadK': 1, 'ThreadN': 16, 'WarpM': 2, 'WarpN': 1, 'TYPE': 'float'}),
    KernelConfig({'BlockM': 32, 'BlockK': 32, 'BlockN': 128, 'TM': 4, 'TN': 4, 'TK': 2, 'ThreadM': 2, 'ThreadK': 1, 'ThreadN': 16, 'WarpM': 4, 'WarpN': 2, 'TYPE': 'float'}),
    KernelConfig({'BlockM': 16, 'BlockK': 32, 'BlockN': 64, 'TM': 4, 'TN': 4, 'TK': 1, 'ThreadM': 2, 'ThreadK': 1, 'ThreadN': 16, 'WarpM': 2, 'WarpN': 1, 'TYPE': 'float'}),
    KernelConfig({'BlockM': 32, 'BlockK': 32, 'BlockN': 128, 'TM': 4, 'TN': 4, 'TK': 1, 'ThreadM': 2, 'ThreadK': 1, 'ThreadN': 16, 'WarpM': 4, 'WarpN': 2, 'TYPE': 'float'}),
    KernelConfig({'BlockM': 32, 'BlockK': 32, 'BlockN': 128, 'TM': 4, 'TN': 4, 'TK': 1, 'ThreadM': 2, 'ThreadK': 1, 'ThreadN': 16, 'WarpM': 2, 'WarpN': 2, 'TYPE': 'float'}),

    KernelConfig({'BlockM': 64, 'BlockK': 32, 'BlockN': 64, 'TM': 4, 'TN': 4, 'TK': 2, 'ThreadM': 4, 'ThreadK': 1, 'ThreadN': 8, 'WarpM': 4, 'WarpN': 2, 'TYPE': 'float'}),
    KernelConfig({'BlockM': 32, 'BlockK': 32, 'BlockN': 32, 'TM': 4, 'TN': 4, 'TK': 1, 'ThreadM': 4, 'ThreadK': 1, 'ThreadN': 8, 'WarpM': 2, 'WarpN': 1, 'TYPE': 'float'}),
    KernelConfig({'BlockM': 32, 'BlockK': 32, 'BlockN': 64, 'TM': 4, 'TN': 4, 'TK': 1, 'ThreadM': 4, 'ThreadK': 1, 'ThreadN': 8, 'WarpM': 2, 'WarpN': 2, 'TYPE': 'float'}),
    KernelConfig({'BlockM': 16, 'BlockK': 32, 'BlockN': 64, 'TM': 4, 'TN': 4, 'TK': 1, 'ThreadM': 4, 'ThreadK': 1, 'ThreadN': 8, 'WarpM': 1, 'WarpN': 2, 'TYPE': 'float'}),
    KernelConfig({'BlockM': 64, 'BlockK': 32, 'BlockN': 64, 'TM': 4, 'TN': 4, 'TK': 1, 'ThreadM': 4, 'ThreadK': 1, 'ThreadN': 8, 'WarpM': 4, 'WarpN': 2, 'TYPE': 'float'}),
    KernelConfig({'BlockM': 32, 'BlockK': 32, 'BlockN': 128, 'TM': 4, 'TN': 4, 'TK': 1, 'ThreadM': 4, 'ThreadK': 1, 'ThreadN': 8, 'WarpM': 2, 'WarpN': 4, 'TYPE': 'float'}),

    KernelConfig({'BlockM': 32, 'BlockK': 32, 'BlockN': 32, 'TM': 4, 'TN': 4,  'TK': 2, 'ThreadM': 4, 'ThreadK': 1, 'ThreadN': 8, 'WarpM': 2, 'WarpN': 1, 'TYPE': 'float'}),
    KernelConfig({'BlockM': 32, 'BlockK': 32, 'BlockN': 64, 'TM': 4, 'TN': 4,  'TK': 2, 'ThreadM': 4, 'ThreadK': 1, 'ThreadN': 8, 'WarpM': 2, 'WarpN': 2, 'TYPE': 'float'}),
    KernelConfig({'BlockM': 16, 'BlockK': 32, 'BlockN': 64, 'TM': 4, 'TN': 4,  'TK': 2, 'ThreadM': 4, 'ThreadK': 1, 'ThreadN': 8, 'WarpM': 1, 'WarpN': 2, 'TYPE': 'float'}),
    KernelConfig({'BlockM': 64, 'BlockK': 32, 'BlockN': 64, 'TM': 4, 'TN': 4,  'TK': 2, 'ThreadM': 4, 'ThreadK': 1, 'ThreadN': 8, 'WarpM': 4, 'WarpN': 2, 'TYPE': 'float'}),
    KernelConfig({'BlockM': 32, 'BlockK': 32, 'BlockN': 128, 'TM': 4, 'TN': 4, 'TK': 2, 'ThreadM': 4, 'ThreadK': 1, 'ThreadN': 8, 'WarpM': 2, 'WarpN': 4, 'TYPE': 'float'}),

]

kernel_simt = KernelHandler(
    source_file='src/simt_gemm.cu',
    compile_configs=hand_picked_configs,
    keys=['m', 'k', 'n'],
    platform='nvidia',
    disable_benchmark=False,
    ignore_compile_errors=True,
    parallel_compile=True
)

def simt_gemm(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    assert a.shape[1] == b.shape[0]
    assert len(a.shape) == len(b.shape) == 2
    m, k = a.shape
    n = b.shape[1]
    c = torch.empty((m, n), device=a.device, dtype=a.dtype)
    kernel_simt(a, b, c, m=m, k=k, n=n)
    return c


if __name__ == '__main__':
    a = torch.arange(0, 64, device='cuda')[None, :] * 64 + torch.arange(0, 64, device='cuda')[:, None]
    a = a.float()
    a = torch.randn([1024, 1024], device='cuda')
    b = torch.randn([1024, 1024], device='cuda')
    c1 = a @ b
    c = simt_gemm(a, b)
    err = (c1 - c).abs()
    print(c)
    print(err)
    print(err.max())

    a = torch.randn([512, 512], device='cuda')
    b = torch.randn([512, 512], device='cuda')
    c1 = a @ b
    c = rocgemm(a, b)

    err = (a @ b - c).abs()
    print(err.max())

