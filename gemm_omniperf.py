# %%
import os
print(os.getcwd())
import torch
from kernels.gemm_hidet import hidet_simt
from kernels.composable_kernel_gemm import ck_gemm, ck_gemm_dl
from kernels.gemm_mfmav1 import mfma_gemmv1
from kernels.gemm_mfmav2 import mfma_gemmv2
from kernels.gemm_mfmav3 import mfma_gemmv3, mfma_gemmv3_5
from kernels.rocwmma_gemm import wmma_gemm
from triton.ops.matmul import matmul

a = torch.empty([2048, 2048], device='cuda')
b = torch.empty([2048, 2048], device='cuda')

# c = a @ b
# hidet_simt(a, b, version=1)
# ck_gemm(a, b, 22)
# ck_gemm_dl(a, b)
# mfma_gemmv1(a, b, version=1)
# mfma_gemmv2(a, b)
# mfma_gemmv2(a, b, ver=1, pack_len=4)
mfma_gemmv3(a, b, ver=2) #, so_name='_BLOCK_K=16__BLOCK_M=128__BLOCK_N=64__InnerK=16__VecLoad=1__Warps=4.so')
# mfma_gemmv3(a, b, ver=4)
# mfma_gemmv3(a, b, ver=5) #, so_name='_BLOCK_K=16__BLOCK_M=128__BLOCK_N=64__InnerK=16__VecLoad=1__Warps=4.so')
# mfma_gemmv3_5(a, b)
# mfma_gemmv3(a, b, ver=6)

# wmma_gemm(a, b)
# matmul(a, b)

# %%
