# # %%
# import math
# import torch
# from kernels import KernelHandler, KernelConfig

# kernel = KernelHandler(
#     source_file='src/saxpy.cu', 
#     compile_configs=[
#         KernelConfig({'BLOCKSIZE': 512, 'REPEATS': 4}),
#         KernelConfig({'BLOCKSIZE': 256, 'REPEATS': 8}),
#         KernelConfig({'BLOCKSIZE': 1024, 'REPEATS': 2}),
#         KernelConfig({'BLOCKSIZE': 1024, 'REPEATS': 4}),
#     ], 
#     keys=['n'], 
#     platform='nvidia'
# )

# def saxpy(a: torch.Tensor, x: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
#     assert a.shape == b.shape == x.shape
#     n = math.prod(a.shape)
#     c = torch.empty_like(a)
#     kernel(a, x, b, c, n=n)
#     return c

# if __name__ == '__main__':
#     a = torch.randn(1000, device='cuda')
#     x = torch.randn(1000, device='cuda')
#     b = torch.randn(1000, device='cuda')
#     c = saxpy(a, x, b)
#     c1 = a * x + b
#     print(torch.allclose(c, c1))

#     a = torch.randn(2, 1000, device='cuda')
#     b = torch.randn(2, 1000, device='cuda')
#     x = torch.randn(2, 1000, device='cuda')
#     c = torch.zeros_like(a)
#     c = saxpy(a, x, b)
#     c1 = a * x + b
#     print(torch.allclose(c, c1))

