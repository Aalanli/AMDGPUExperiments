# %%
import torch
print(torch.cuda.is_available())

a = torch.randn([512, 512], device='cuda')
b = torch.rand_like(a)

c = a @ b

import triton
import triton.language as tl

from triton import compile
@triton.jit
def test(at, bt, ct):
    midx = tl.arange(0, 32)
    kidx = tl.arange(0, 32)
    nidx = tl.arange(0, 32)

    aidx = midx[:, None] * 32 + kidx[None, :]
    bidx = kidx[:, None] * 32 + nidx[None, :]
    cidx = midx[:, None] * 32 + nidx[None, :]

    a = tl.load(at + aidx)
    b = tl.load(bt + bidx)
    
    x = tl.dot(a, b)
    tl.atomic_add(ct + cidx, x)

a = torch.randn([32, 32], device='cuda')
b = torch.randn([32, 32], device='cuda')
c = torch.zeros([32, 32], device='cuda')
test[(1,)](a, b, c)

kernel = compile(test, signature='*fp32,*fp32,*fp32')
print(kernel.asm['amdgcn'])

