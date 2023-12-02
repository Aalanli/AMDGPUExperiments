# %%
import torch
from kernels import KernelHandler, KernelConfig, PLATFORM


def gen_configs():
    if PLATFORM == 'nvidia':
        block_warps = [(4, 8), (2, 16), (16, 2), (8, 4)]
    else:
        block_warps = [(4, 16), (16, 4), (8, 8)]
    for (bwm, bwn) in [(1, 1), (2, 2), (1, 2), (2, 1), (2, 4), (4, 2)]:
        for (wm, wn) in block_warps:
            for (wom, won) in [(1, 1), (1, 2), (2, 2), (1, 3), (3, 1), (2, 3), (3, 2), (3, 3)]:
                for bk in [4, 8, 16]:
                    for (tm, tn) in [(4, 4)]:
                        block_m = bwm * wom * wm * tm
                        block_n = bwn * won * wn * tn
                        block_k = bk
                        smem = (block_m * block_k + block_k * block_n) * 4 * 2
                        regs = (wom * tm + won * tn) * 2 + wom * won * tm * tn * 2
                        if PLATFORM == 'nvidia':
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

if PLATFORM == 'nvidia':
    hand_picked_configs = [
        KernelConfig({
            'BlockWarpsK': 4, 'BlockWarpsM': 2, 'BlockWarpsN': 2, 
            'WarpOuterM': 2, 'WarpOuterN': 2, 'WarpMidM': 4, 
            'WarpMidN': 8, 'WarpInnerM': 4, 'WarpInnerN': 4
        }),
    ]
else:
    hand_picked_configs = [
        # KernelConfig({
        #     'BlockWarpsK': 4, 'BlockWarpsM': 2, 'BlockWarpsN': 2, 
        #     'WarpOuterM': 2, 'WarpOuterN': 2, 'WarpMidM': 8, 
        #     'WarpMidN': 8, 'WarpInnerM': 4, 'WarpInnerN': 4
        # }),
        KernelConfig({
            'BlockWarpsK': 8, 'BlockWarpsM': 2, 'BlockWarpsN': 4, 
            'WarpOuterM': 1, 'WarpOuterN': 1, 'WarpMidM': 8, 
            'WarpMidN': 8, 'WarpInnerM': 4, 'WarpInnerN': 4
        }),
    ]

source_file = 'src/simt_gemm/simt_gemm_hidet.cu' if PLATFORM == 'nvidia' else 'src/simt_gemm/simt_gemm_hidet.cpp'

hidet_kernel = KernelHandler(
    source_file=source_file,
    compile_configs=list(gen_configs()),
    keys=['m', 'k', 'n', 'version'],
    platform=PLATFORM,
    disable_benchmark=False,
    ignore_compile_errors=True,
    parallel_compile=True,
    arch='gfx90a',
)


def hidet_simt(a: torch.Tensor, b: torch.Tensor, version=0) -> torch.Tensor:
    assert a.shape[1] == b.shape[0]
    assert len(a.shape) == len(b.shape) == 2
    m, k = a.shape
    n = b.shape[1]
    c = torch.empty((m, n), device=a.device, dtype=a.dtype)
    hidet_kernel(a, b, c, m=m, k=k, n=n, version=version)
    return c


if __name__ == '__main__':
    d = 64
    a = torch.arange(0, 64, device='cuda')[None, :] * 64 + torch.arange(0, 64, device='cuda')[:, None]
    a = a.float()
    a = torch.randn([1972, 1972], device='cuda')
    b = torch.randn([1972, 1972], device='cuda')
    c1 = a @ b
    c = hidet_simt(a, b, 2)
    err = (c1 - c).abs()
    print(c)
    print(err)
    print(err.max())

