# %%
import torch
from kernels import KernelHandler, KernelConfig

def gen_configs():
    for (bwm, bwn) in [(1, 1), (2, 2), (1, 2), (2, 1), (2, 4), (4, 2)]:
        for (wm, wn) in [(4, 8), (2, 16), (16, 2), (8, 4)]:
            for (wom, won) in [(1, 1), (1, 2), (2, 2), (1, 3), (3, 1), (2, 3), (3, 2), (3, 3)]:
                for (bkm, bkn) in [(4, 4), (8, 4), (8, 8), (16, 4), (32, 4), (16, 8)]:
                    for (tm, tn) in [(4, 4)]:
                        block_m = bwm * wom * wm * tm
                        block_n = bwn * won * wn * tn
                        smem = (block_m * bkm + block_n * bkn) * 4 * 2
                        regs = (wom * tm + won * tn) * 2 + wom * won * tm * tn * 2
                        if regs > 255:
                            continue
                        if smem > 0xc000:
                            continue
                        nthreads = wm * wn * bwm * bwn
                        warp_size = wm * wn
                        if nthreads % bkm != 0: continue
                        if nthreads % bkn != 0: continue
                        if warp_size % bkm != 0: continue
                        if warp_size % bkn != 0: continue
                        stride_ldgn = nthreads // bkn
                        stride_ldgm = nthreads // bkm
                        if stride_ldgn > block_n: continue
                        if stride_ldgm > block_m: continue
                        
                        yield KernelConfig({
                            'BlocksizeK_M': bkm,
                            'BlocksizeK_N': bkn,
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
        'BlocksizeK_M': 4, 'BlocksizeK_N': 4,
        'BlockWarpsM': 2, 'BlockWarpsN': 2, 
        'WarpOuterM': 2, 'WarpOuterN': 2, 
        'WarpMidM': 4, 'WarpMidN': 8, 
        'WarpInnerM': 4, 'WarpInnerN': 4
    }),
    KernelConfig({
        'BlocksizeK_M': 8, 'BlocksizeK_N': 4,
        'BlockWarpsM': 1, 'BlockWarpsN': 2, 
        'WarpOuterM': 2, 'WarpOuterN': 2, 
        'WarpMidM': 4, 'WarpMidN': 8, 
        'WarpInnerM': 4, 'WarpInnerN': 4
    }),
]

# print(any(hand_picked_configs[0].config == c.config for c in gen_configs()))


kernel_simtv6 = KernelHandler(
    source_file='src/simt_gemmv6.cu',
    compile_configs=hand_picked_configs,
    keys=['m', 'k', 'n'],
    platform='nvidia',
    disable_benchmark=False,
    ignore_compile_errors=True,
    parallel_compile=True
)


def simt_gemmv6(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    assert a.shape[1] == b.shape[0]
    assert len(a.shape) == len(b.shape) == 2
    m, k = a.shape
    n = b.shape[1]
    c = torch.empty((m, n), device=a.device, dtype=a.dtype)
    kernel_simtv6(a, b, c, m=m, k=k, n=n)
    return c


if __name__ == '__main__':
    d = 64
    a = torch.arange(0, 64, device='cuda')[None, :] * 64 + torch.arange(0, 64, device='cuda')[:, None]
    a = a.float()
    a = torch.randn([1024, 1024], device='cuda')
    b = torch.randn([1024, 1024], device='cuda')
    c1 = a @ b
    c = simt_gemmv6(a, b)
    err = (c1 - c).abs()
    print(c)
    print(err)
    print(err.max())

