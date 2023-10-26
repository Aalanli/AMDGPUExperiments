# %%
from typing import List, Dict, Any
import hashlib
import os
import json
import shutil
import subprocess
import ctypes

def build(source, out_path, **kwargs):
    args = [f'-D {k}={v}' for k, v in kwargs.items()]
    assert os.path.exists(source) and os.path.isfile(source)

    subprocess.run(['hipcc', '-fPIC', '-O3', '-c', source] + args + ['-o', out_path], check=True)
    subprocess.run(['hipcc', '-shared', '-o', out_path, out_path], check=True)

build('saxpy.cpp', 'saxpy.so', BLOCKSIZE=512, REPEATS=4, LAUNCH_NAME='launch10')

def is_fundamental_type(a):
    return isinstance(a, (int, float, str, bool))

CACHE_DIR = '~/.cache/hip_kernels/'

class KernelConfig:
    def __init__(self, config: Dict[str, Any], key: List[str]):
        self.config = config
        self.key = key
        self.key.sort()
        for k in key:
            assert k in self.config, f'Key {k} not in config'
        for k, v in config.items():
            assert is_fundamental_type(v), f'Key {k}: {v} is not a fundamental type'

    def so_name(self) -> str:
        name = ''
        sorted_config = list(self.config.items())
        sorted_config.sort(key=lambda x: x[0])
        for k, v in sorted_config:
            name += f'{k}={v}_'
        return name[:-1] + '.so'
    
    def launch_name(self) -> str:
        name = ''
        sorted_config = list(self.config.items())
        sorted_config.sort(key=lambda x: x[0])
        for k, v in sorted_config:
            res = str(v)
            res = filter(lambda x: x.isalnum(), res)
            res = ''.join(res)
            name += f'{k}_{res}_'
        return 'launch_' + name[:-1]
    
    def key_name(self, runtime_args: Dict[str, Any]) -> str:
        name = ''
        for k in self.key:
            assert is_fundamental_type(runtime_args[k]), f'Key {k}: {runtime_args[k]} is not a fundamental type'
            name += f'{k}={runtime_args[k]}_'
        return name[:-1]


class KernelHandler:
    def __init__(self, source_file: str, compile_configs: List[KernelConfig]):
        # CACHE_DIR
        # |_ hash(source_file)
        # |__ fmt(config[0]).so
        # |__ fmt(config[1]).so
        # |__ ...
        # |__ fmt(config[n]).so
        # |__ source_file
        # |__ meta_data.json
        self.source_file = source_file
        assert os.path.exists(self.source_file) and os.path.isfile(self.source_file), f'File {self.source_file} does not exist'
        file_name = os.path.basename(self.source_file)
        file_name_no_ext = os.path.splitext(file_name)[0]
        with open(self.source_file, 'r') as f:
            self.source = f.read()
        folderhash = hashlib.sha256(self.source.encode('utf-8')).hexdigest()
        dir_path = os.path.join(CACHE_DIR, file_name_no_ext + '_' + folderhash[:10])
        self.dir_path = dir_path

        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

        self.so_name2_config: Dict[str, KernelConfig] = {
            config.so_name(): config for config in compile_configs
        }
        need_to_compile = filter(
            lambda x: not os.path.exists(os.path.join(dir_path, x)), self.so_name2_config.keys()
        )
        for so_name in need_to_compile:
            config = self.so_name2_config[so_name]
            launch_name = config.launch_name()
            build(self.source_file, os.path.join(dir_path, so_name), LAUNCH_NAME=launch_name, **config)
        
        libraries = {}
        
        shutil.copy(self.source_file, dir_path)

        # map from runtime-key to so_name
        self.kernel_map: Dict[str, str] = {}
        if os.path.exists(os.path.join(dir_path, 'meta_data.json')):
            with open(os.path.join(dir_path, 'meta_data.json'), 'r') as f:
                self.kernel_map = json.load(f)            
    
    def __call__(self, **kwargs):
        runtime_args = kwargs
    


# %%
import ctypes
import torch
def saxpy(a: torch.Tensor, b: torch.Tensor):
    assert a.shape == b.shape
    assert a.dtype == b.dtype == torch.float32
    c = torch.empty_like(a)
    lib = ctypes.cdll.LoadLibrary('saxpy.so')
    
    lib.launch10(a.data_ptr(), b.data_ptr(), c.data_ptr(), a.numel())

a = torch.randn(1000, device='cuda')
b = torch.randn(1000, device='cuda')
c = torch.empty_like(a)

# %%
lib = ctypes.cdll.LoadLibrary('saxpy.so')
lib.launch10()
