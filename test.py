# %%
contig = 2
a, b = 4, 16
for i in range(0, a):
    if a < b // contig:
        print([(i ^ (j // contig)) * contig for j in range(0, b)])
    else:
        print([((i % (b // contig)) ^ (j // contig)) * contig for j in range(0, b)])


# %%
import torch
from kernels.rocwmma_gemm import wmma_gemm

if __name__ == '__main__':
    a = torch.arange(0, 64, device='cuda')[None, :] + torch.arange(0, 64, device='cuda')[:, None] * 64
    a = a.float()
    a = torch.randn([512, 512], device='cuda')
    b = torch.randn([512, 512], device='cuda')
    # b = torch.eye(64, device='cuda')
    c1 = a @ b
    c = wmma_gemm(a, b)
    err = (c1 - c).abs()
    print(c)
    print(err)
    print(err.max())


exit()
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
def test(at, bt, ct, k):
    midx = tl.arange(0, 32)
    kidx = tl.arange(0, 32)
    nidx = tl.arange(0, 32)

    aidx = midx[:, None] * 32 + kidx[None, :]
    bidx = kidx[:, None] * 32 + nidx[None, :]
    cidx = midx[:, None] * 32 + nidx[None, :]

    a_ptrs = at + aidx
    b_ptrs = bt + bidx
    c_ptrs = ct + cidx
    for i in range(k):
        a = tl.load(a_ptrs)
        b = tl.load(b_ptrs)
        x = tl.dot(a, b)
        tl.atomic_add(c_ptrs, x)
        a_ptrs += 32
        b_ptrs += 32
        c_ptrs += 32

# a = torch.randn([32, 32], device='cuda')
# b = torch.randn([32, 32], device='cuda')
# c = torch.zeros([32, 32], device='cuda')
# test[(1,)](a, b, c)

kernel = compile(test, signature='*fp32,*fp32,*fp32,i32')
print(kernel.asm['amdgcn'])

