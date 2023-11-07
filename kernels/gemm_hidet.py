# %%
import torch
from kernels import KernelHandler, KernelConfig


def gen_configs():
    for (bwm, bwn) in [(1, 1), (2, 2), (1, 2), (2, 1), (2, 4), (4, 2)]:
        for (wm, wn) in [(4, 8), (2, 16), (16, 2), (8, 4)]:
            for (wom, won) in [(1, 1), (1, 2), (2, 2), (1, 3), (3, 1), (2, 3), (3, 2), (3, 3)]:
                for bk in [4, 8, 16]:
                    for (tm, tn) in [(4, 4)]:
                        block_m = bwm * wom * wm * tm
                        block_n = bwn * won * wn * tn
                        block_k = bk
                        smem = (block_m * block_k + block_k * block_n) * 4 * 2
                        regs = (wom * tm + won * tn) * 2 + wom * won * tm * tn * 2
                        if regs > 255:
                            continue
                        if smem > 0xc000:
                            continue
                        nthreads = wm * wn * bwm * bwn
                        if block_n % (nthreads // block_k) != 0:
                            continue
                        if block_m % (nthreads // block_k) != 0:
                            continue
                        yield KernelConfig({
                            'BlockWarpsK': bk,
                            'BlockWarpsM': bwm,
                            'BlockWarpsN': bwn,
                            'WarpOuterM': wom,
                            'WarpOuterN': won,
                            'WarpMidM': wm,
                            'WarpMidN': wn,
                            'WarpInnerM': tm,
                            'WarpInnerN': tn,
                        })

hand_picked_configs = [
    KernelConfig({
        'BlockWarpsK': 4, 'BlockWarpsM': 2, 'BlockWarpsN': 2, 
        'WarpOuterM': 2, 'WarpOuterN': 2, 'WarpMidM': 4, 
        'WarpMidN': 8, 'WarpInnerM': 4, 'WarpInnerN': 4
    }),
]

hidet_kernel = KernelHandler(
    source_file='src/simt_gemm_hidet.cu',
    compile_configs=list(gen_configs()),
    keys=['m', 'k', 'n'],
    platform='nvidia',
    disable_benchmark=False,
    ignore_compile_errors=True,
    parallel_compile=True
)


def hidet_simt(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    assert a.shape[1] == b.shape[0]
    assert len(a.shape) == len(b.shape) == 2
    m, k = a.shape
    n = b.shape[1]
    c = torch.empty((m, n), device=a.device, dtype=a.dtype)
    hidet_kernel(a, b, c, m=m, k=k, n=n)
    return c


if __name__ == '__main__':
    d = 64
    a = torch.arange(0, 64, device='cuda')[None, :] * 64 + torch.arange(0, 64, device='cuda')[:, None]
    a = a.float()
    a = torch.randn([1024, 1024], device='cuda')
    b = torch.randn([1024, 1024], device='cuda')
    c1 = a @ b
    c = hidet_simt(a, b)
    err = (c1 - c).abs()
    print(c)
    print(err)
    print(err.max())

