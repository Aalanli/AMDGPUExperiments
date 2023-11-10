# %%
from .build_kernel import KernelHandler, KernelConfig, do_bench
from .utils import Bench, BenchData
# from .saxpy import saxpy

import torch
PLATFORM = 'amd' if torch.version.hip is not None else 'nvidia'
