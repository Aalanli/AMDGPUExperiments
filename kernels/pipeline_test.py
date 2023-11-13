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
        kernels[param] = KernelHandler('src/pipeline_test.cpp', 
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


# sanity check, increase arithmetic intensity
def sanity_check_r16(i, **kwargs):
    param = PipelineParam(reg_iter=i)
    run_experiment(param, grid_dim=4, repeats=16, additional_smem=0)
def sanity_check_r32(i, **kwargs):
    param = PipelineParam(reg_iter=i)
    run_experiment(param, grid_dim=4, repeats=32, additional_smem=0)

bench = Bench(x_vals=[1, 2, 4, 8, 16, 32, 64], x_name='fma per register')
bench.bench(sanity_check_r16)
bench.bench(sanity_check_r32)
data = bench.run()
data.show_plot()

