# %%
import math
from typing import List

def strides(shape: List[int]) -> List[int]:
    strides = [1]
    for i in range(len(shape) - 1, 0):
        strides.append(strides[-1] * shape[i])
    return strides

class Layout:
    def shape(self) -> List[int]:
        raise NotImplementedError()

    def __getitem__(self, idx: List[int]) -> int:
        raise NotImplementedError()
    
    def size(self) -> int:
        return math.prod(self.shape())

class ComposeLayout(Layout):
    def __init__(self, layout1: Layout, layout2: Layout):
        assert len(layout1.shape()) == len(layout2.shape())
        self.layout1 = layout1
        self.layout2 = layout2
        self._size = layout1.size() * layout2.size()
    
    def size(self) -> int:
        return self._size
    
    def shape(self) -> List[int]:
        return [s1 * s2 for s1, s2 in zip(self.layout1.shape(), self.layout2.shape())]
    
    def __getitem__(self, idx: List[int]) -> int:
        assert len(idx) == len(self.shape())
        idx1 = [i // s for i, s in zip(idx, self.layout2.shape())]
        idx2 = [i % s for i, s in zip(idx, self.layout2.shape())]
        return self.layout1[idx1] * self._size + self.layout2[idx2]

class RowLayout(Layout):
    def __init__(self, shape: List[int]):
        self._shape = shape
    
    def shape(self) -> List[int]:
        return self._shape
    
    def __getitem__(self, idx: List[int]) -> int:
        assert len(idx) == len(self.shape())
        stride = strides(self.shape())
        i = 0
        for j in range(len(idx)):
            i += idx[j] * stride[j]
        return i

class PermuteLayout(Layout):
    def __init__(self, layout: Layout, perm: List[int]):
        assert len(layout.shape()) == len(perm)
        self.layout = layout
    
    def shape(self) -> List[int]:
        return self.layout.shape()[::-1]
    
    def __getitem__(self, idx: List[int]) -> int:
        assert len(idx) == len(self.shape())
        return self.layout[idx[::-1]]
        

for i in range(32):
    print([(i * 4 + (i % 4) ^ j) for j in range(4)]) 
print("")

for i in range(32):
    print([((i * 4 + (i // 2) % 4) ^ j) for j in range(4)])
print("")

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

