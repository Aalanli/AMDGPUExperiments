# %%
import os
print(os.getcwd())
import torch
from kernels.gemm_hidet import hidet_simt
from kernels.composable_kernel_gemm import ck_gemm, ck_gemm_dl
from kernels.gemm_mfmav1 import mfma_gemmv1
from kernels.gemm_mfmav2 import mfma_gemmv2
from kernels.rocwmma_gemm import wmma_gemm
from triton.ops.matmul import matmul

a = torch.empty([1024, 1024], device='cuda')
b = torch.empty([1024, 1024], device='cuda')

c = a @ b
hidet_simt(a, b, version=1)
ck_gemm(a, b, 14)
ck_gemm_dl(a, b)
mfma_gemmv1(a, b)
mfma_gemmv2(a, b)
wmma_gemm(a, b)
# matmul(a, b)
