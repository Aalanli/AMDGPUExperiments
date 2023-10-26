# %%

import os

import subprocess

def build(source, out_path, **kwargs):
    args = [f'-D {k}={v}' for k, v in kwargs.items()]
    assert os.path.exists(source) and os.path.isfile(source)

    subprocess.run(['hipcc', '-fPIC', '-O3', '-c', source] + args + ['-o', out_path], check=True)
    subprocess.run(['hipcc', '-shared', '-o', out_path, out_path], check=True)


build('saxpy.cpp', 'saxpy.so', BLOCKSIZE=512, REPEATS=4, LAUNCH_NAME='launch10')

# %%
import ctypes
import torch

a = torch.randn(1000, device='cuda')
b = torch.randn(1000, device='cuda')
c = torch.empty_like(a)

# %%
lib = ctypes.cdll.LoadLibrary('saxpy.so')
lib.launch10()
