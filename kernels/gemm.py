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
    keys=['m', 'n', 'k'],
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


kernel_simt = KernelHandler(
    source_file='src/simt_gemm.cpp',
    compile_configs=[
        KernelConfig({'BlockM': 128, 'BlockK': 32, 'BlockN': 64,  'WarpM': 4, 'WarpN': 2, 'ThreadM': 8, 'ThreadK': 1, 'ThreadN': 4, 'TM': 4, 'TN': 4, 'TK': 4, 'TYPE': 'float'}),
        KernelConfig({'BlockM': 128, 'BlockK': 32, 'BlockN': 64,  'WarpM': 2, 'WarpN': 2, 'ThreadM': 8, 'ThreadK': 1, 'ThreadN': 4, 'TM': 4, 'TN': 4, 'TK': 4, 'TYPE': 'float'}),
        KernelConfig({'BlockM': 64,  'BlockK': 32, 'BlockN': 128, 'WarpM': 2, 'WarpN': 4, 'ThreadM': 8, 'ThreadK': 1, 'ThreadN': 4, 'TM': 4, 'TN': 4, 'TK': 4, 'TYPE': 'float'}),
        KernelConfig({'BlockM': 64,  'BlockK': 32, 'BlockN': 128, 'WarpM': 2, 'WarpN': 2, 'ThreadM': 8, 'ThreadK': 1, 'ThreadN': 4, 'TM': 4, 'TN': 4, 'TK': 4, 'TYPE': 'float'}),

        KernelConfig({'BlockM': 128, 'BlockK': 32, 'BlockN': 32,  'WarpM': 4, 'WarpN': 2, 'ThreadM': 8, 'ThreadK': 1, 'ThreadN': 4, 'TM': 4, 'TN': 4, 'TK': 4, 'TYPE': 'float'}),
        KernelConfig({'BlockM': 64,  'BlockK': 32, 'BlockN': 64,  'WarpM': 2, 'WarpN': 4, 'ThreadM': 8, 'ThreadK': 1, 'ThreadN': 4, 'TM': 4, 'TN': 4, 'TK': 4, 'TYPE': 'float'}),
        KernelConfig({'BlockM': 64,  'BlockK': 32, 'BlockN': 32,  'WarpM': 2, 'WarpN': 2, 'ThreadM': 8, 'ThreadK': 1, 'ThreadN': 4, 'TM': 4, 'TN': 4, 'TK': 4, 'TYPE': 'float'}),
        # KernelConfig({'BlockM': 64,  'BlockK': 16, 'BlockN': 128, 'WarpM': 2, 'WarpN': 4, 'ThreadM': 8, 'ThreadK': 1, 'ThreadN': 4, 'TM': 4, 'TN': 4, 'TK': 4, 'TYPE': 'float'}),
        # KernelConfig({'BlockM': 64,  'BlockK': 16, 'BlockN': 128, 'WarpM': 2, 'WarpN': 2, 'ThreadM': 8, 'ThreadK': 1, 'ThreadN': 4, 'TM': 4, 'TN': 4, 'TK': 4, 'TYPE': 'float'}),

        # KernelConfig({'BlockM': 64,  'BlockK': 16, 'BlockN': 64,  'WarpM': 2, 'WarpN': 2, 'ThreadM': 8, 'ThreadK': 1, 'ThreadN': 4, 'TM': 4, 'TN': 4, 'TK': 4, 'TYPE': 'float'}),
        KernelConfig({'BlockM': 64,  'BlockK': 32, 'BlockN': 64,  'WarpM': 2, 'WarpN': 2, 'ThreadM': 8, 'ThreadK': 1, 'ThreadN': 4, 'TM': 4, 'TN': 4, 'TK': 4, 'TYPE': 'float'}),
        KernelConfig({'BlockM': 64,  'BlockK': 64, 'BlockN': 64,  'WarpM': 2, 'WarpN': 2, 'ThreadM': 8, 'ThreadK': 1, 'ThreadN': 4, 'TM': 4, 'TN': 4, 'TK': 4, 'TYPE': 'float'}),
        # KernelConfig({'BlockM': 64,  'BlockK': 128,'BlockN': 64,  'WarpM': 2, 'WarpN': 2, 'ThreadM': 8, 'ThreadK': 1, 'ThreadN': 4, 'TM': 4, 'TN': 4, 'TK': 4, 'TYPE': 'float'}),

        # KernelConfig({'BlockM': 64,  'BlockK': 16, 'BlockN': 64,  'WarpM': 2, 'WarpN': 4, 'ThreadM': 8, 'ThreadK': 1, 'ThreadN': 4, 'TM': 4, 'TN': 4, 'TK': 4, 'TYPE': 'float'}),
        KernelConfig({'BlockM': 64,  'BlockK': 32, 'BlockN': 32,  'WarpM': 2, 'WarpN': 2, 'ThreadM': 8, 'ThreadK': 1, 'ThreadN': 4, 'TM': 4, 'TN': 4, 'TK': 4, 'TYPE': 'float'}),
        KernelConfig({'BlockM': 64,  'BlockK': 64, 'BlockN': 32,  'WarpM': 2, 'WarpN': 2, 'ThreadM': 8, 'ThreadK': 1, 'ThreadN': 4, 'TM': 4, 'TN': 4, 'TK': 4, 'TYPE': 'float'}),
        # KernelConfig({'BlockM': 64,  'BlockK': 128,'BlockN': 64,  'WarpM': 2, 'WarpN': 4, 'ThreadM': 8, 'ThreadK': 1, 'ThreadN': 4, 'TM': 4, 'TN': 4, 'TK': 4, 'TYPE': 'float'}),

        # KernelConfig({'BlockM': 64,  'BlockK': 16, 'BlockN': 64,  'WarpM': 2, 'WarpN': 2, 'ThreadM': 8, 'ThreadK': 1, 'ThreadN': 4, 'TM': 4, 'TN': 4, 'TK': 4, 'TYPE': 'float'}),
        KernelConfig({'BlockM': 64,  'BlockK': 32, 'BlockN': 64,  'WarpM': 2, 'WarpN': 2, 'ThreadM': 8, 'ThreadK': 1, 'ThreadN': 4, 'TM': 4, 'TN': 4, 'TK': 4, 'TYPE': 'float'}),
        KernelConfig({'BlockM': 64,  'BlockK': 64, 'BlockN': 64,  'WarpM': 2, 'WarpN': 2, 'ThreadM': 8, 'ThreadK': 1, 'ThreadN': 4, 'TM': 4, 'TN': 4, 'TK': 4, 'TYPE': 'float'}),
        # KernelConfig({'BlockM': 64,  'BlockK': 128,'BlockN': 64,  'WarpM': 2, 'WarpN': 2, 'ThreadM': 8, 'ThreadK': 1, 'ThreadN': 4, 'TM': 4, 'TN': 4, 'TK': 4, 'TYPE': 'float'})
    ],
    keys=['m', 'k', 'n'],
    platform='nvidia',
    disable_benchmark=False
)

def simt_gemm(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    assert a.shape[1] == b.shape[0]
    assert len(a.shape) == len(b.shape) == 2
    m, k = a.shape
    n = b.shape[1]
    c = torch.ones((m, n), device=a.device, dtype=a.dtype) * -1
    kernel_simt(a, b, c, m=m, k=k, n=n)
    return c


if __name__ == '__main__':
    d = 64
    a = torch.arange(0, 64, device='cuda')[None, :] * 64 + torch.arange(0, 64, device='cuda')[:, None]
    a = a.float()
    a = torch.randn([512, 512], device='cuda')
    b = torch.randn([512, 512], device='cuda')
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

    def bench_rocgemm1(i, **kwargs):
        m, k, n = i
        a = torch.randn([m, k], device='cuda')
        b = torch.randn([k, n], device='cuda')
        return lambda: rocgemm(a, b, version=1)
    
    def bench_simtgemm(i, **kwargs):
        m, k, n = i
        a = torch.randn([m, k], device='cuda')
        b = torch.randn([k, n], device='cuda')
        return lambda: simt_gemm(a, b)

    def bench_rocgemm2(i, **kwargs):
        m, k, n = i
        a = torch.randn([m, k], device='cuda')
        b = torch.randn([k, n], device='cuda')
        return lambda: rocgemm(a, b, version=2)

    def bench_blas(i, **kwargs):
        m, k, n = i
        a = torch.randn([m, k], device='cuda')
        b = torch.randn([k, n], device='cuda')
        return lambda: a @ b
    # import hidet
    # def bench_hidet(i, **kwargs):
    #     m, n, k = i
    #     a = hidet.randn([m, k], device='cuda')
    #     b = hidet.randn([k, n], device='cuda')
        
    
    square = [(m, m, m) for m in [32, 64, 128, 256, 512, 768, 1024]]
    mem_bound = [(1, m, m) for m in [32, 64, 128, 256, 512, 768, 1024]]
    bench = Bench(
        x_vals=square,
        x_name='(m, k, n)',
    )
    # bench.bench(bench_rocgemm1)
    bench.bench(bench_rocgemm2, 'naive')
    bench.bench(bench_blas, 'blas')
    bench.bench(bench_simtgemm, 'simt')
    data = bench.run()
    data.show_plot()
