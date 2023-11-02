# %%
from kernels.gemm import simt_gemm, rocgemm
import torch

a = torch.randn([1024, 1024], device='cuda')
b = torch.randn([1024, 1024], device='cuda')

simt_gemm(a, b)
rocgemm(a, b, version=2)

