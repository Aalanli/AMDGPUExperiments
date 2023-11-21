# %%
import torch
from kernels import KernelHandler, KernelConfig

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

kernel_simtv2 = KernelHandler(
    source_file='src/simt_gemm/simt_gemmv2.cu',
    compile_configs=hand_picked_configs,
    keys=['m', 'k', 'n'],
    platform='nvidia',
    disable_benchmark=False,
    ignore_compile_errors=True,
    parallel_compile=True
)


def simt_gemmv2(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    assert a.shape[1] == b.shape[0]
    assert len(a.shape) == len(b.shape) == 2
    m, k = a.shape
    n = b.shape[1]
    c = torch.empty((m, n), device=a.device, dtype=a.dtype)
    kernel_simtv2(a, b, c, m=m, k=k, n=n)
    return c


if __name__ == '__main__':
    import hidet
    hidet.option.debug_cache_tuning()
    hidet.option.search_space(2)
    hidet.option.cache_dir('./outs/simt_matmul')

    d = 64
    a = torch.arange(0, 64, device='cuda')[None, :] * 64 + torch.arange(0, 64, device='cuda')[:, None]
    a = a.float()
    a = torch.randn([512, 512], device='cuda')
    b = torch.randn([512, 512], device='cuda')
    c1 = a @ b
    c = simt_gemmv2(a, b)
    err = (c1 - c).abs()
    print(c)
    print(err)
    print(err.max())

