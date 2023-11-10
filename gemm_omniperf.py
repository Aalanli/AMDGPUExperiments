# %%
import os
print(os.getcwd())
import torch
from kernels.gemm_hidet import hidet_simt
a = torch.empty([512, 512], device='cuda')
b = torch.empty([512, 512], device='cuda')

c = a @ b
hidet_simt(a, b, version=1)
