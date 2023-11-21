# %%
from dataclasses import dataclass
from typing import Dict
import torch
from kernels.utils import Bench

from kernels import KernelHandler, KernelConfig, PLATFORM


@dataclass(frozen=True)
class PipelineParam:
    reg_pipe: int = 1
    smem_pipe: int = 1
    reg_iter: int = 1
    smem_iter: int = 1
    smem_read_stride: int = 1
    smem_write_stride: int = 1
    block_dim: int = 256

kernels: Dict[PipelineParam, KernelHandler] = {}

def run_experiment(param: PipelineParam, grid_dim=1, repeats=16, additional_smem=0):
    if param not in kernels:
        kernels[param] = KernelHandler('src/misc/pipeline_test.cpp', 
                                       [KernelConfig({'SHARED_ITERS': param.smem_iter,
                                                     'REG_ITERS': param.reg_iter,
                                                     'SMEM_READ_STRIDE': param.smem_read_stride,
                                                     'SMEM_WRITE_STRIDE': param.smem_write_stride,
                                                     'NPIPELINE_SMEM': param.smem_pipe,
                                                     'NPIPELINE_REGS': param.reg_pipe,
                                                     'BLOCK_DIM': param.block_dim})],
                                        keys=[], platform=PLATFORM, disable_benchmark=True,
                                        ignore_compile_errors=False, parallel_compile=False)
    a = torch.empty([grid_dim * param.block_dim * repeats], device='cuda', dtype=torch.float32)
    b = torch.empty([grid_dim * param.block_dim], device='cuda', dtype=torch.float32)
    kernels[param](a, b, grid_dim, repeats, additional_smem)

run_experiment(PipelineParam())

# %%
# sanity check, increase arithmetic intensity
def sanity_check(reg_iter, repeats):
    param = PipelineParam(reg_iter=reg_iter)
    return lambda: run_experiment(param, grid_dim=4, repeats=repeats, additional_smem=0)

bench = Bench(x_vals=[1, 2, 4, 8, 16, 32, 64], x_name='fma per register')
# for rep in [16, 32]:
#     bench.bench(lambda i, **kwargs: sanity_check(i, rep, **kwargs), name=f'rep{rep}')
bench.bench(lambda i, **kwargs: sanity_check(i, 16, **kwargs), name=f'rep16')
bench.bench(lambda i, **kwargs: sanity_check(i, 32, **kwargs), name=f'rep32')
bench.bench(lambda i, **kwargs: sanity_check(i, 64, **kwargs), name=f'rep64')

data = bench.run()
data.show_plot()


# %%
# hypothesis: increasing smem pipe should have larger impact on performance
# when arithmetic intensity is higher
bench = Bench(x_vals=[1, 2, 4, 8, 16, 32, 64], x_name='fma per register', repeats=64, grid_dim=4)
# for smem_pipe in [1, 2, 3, 4]:
    # bench.bench(lambda i, **kwargs: lambda: run_experiment(PipelineParam(smem_pipe=smem_pipe, reg_iter=i), **kwargs), name=f'smem_pipe{smem_pipe}')
bench.bench(lambda i, **kwargs: lambda: run_experiment(PipelineParam(smem_pipe=1, reg_iter=i), **kwargs), name=f'smem_pipe{1}')
bench.bench(lambda i, **kwargs: lambda: run_experiment(PipelineParam(smem_pipe=2, reg_iter=i), **kwargs), name=f'smem_pipe{2}')
bench.bench(lambda i, **kwargs: lambda: run_experiment(PipelineParam(smem_pipe=3, reg_iter=i), **kwargs), name=f'smem_pipe{3}')
bench.bench(lambda i, **kwargs: lambda: run_experiment(PipelineParam(smem_pipe=4, reg_iter=i), **kwargs), name=f'smem_pipe{4}')

data = bench.run()
data.show_plot()


# %%

