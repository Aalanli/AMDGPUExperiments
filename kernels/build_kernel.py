# %%
from typing import List, Dict, Any, Optional
from tqdm import tqdm
import hashlib
import os
import json
import shutil
import subprocess
import ctypes
import torch


def build(source, out_path, amd=True, **kwargs):
    args = [f' -D {k}={v} ' for k, v in kwargs.items()]
    args = ''.join(args) + '-fPIC'
    assert os.path.exists(source) and os.path.isfile(source)

    file = os.path.basename(source)
    file_ext = os.path.splitext(file)[1]
    if amd:
        assert file_ext == '.cpp', f'AMD kernel must be a cpp file, got {file_ext}'
        subprocess.run(['hipcc', '-O3', '-c', source, args, '-o', out_path, '-I', 'include/'], check=True)
        subprocess.run(['hipcc', '-shared', '-o', out_path, out_path], check=True)
    elif file_ext == '.cpp':
        env = os.environ.copy()
        env['HIP_PLATFORM'] = 'nvidia'
        subprocess.run(['hipcc', '-O3', '-c', source, '--compiler-options', args, '-o', out_path, '-I', 'include/'], check=True, env=env)
        subprocess.run(['hipcc', '-shared', '-o', out_path, out_path], check=True)
    elif file_ext == '.cu':
        subprocess.run(['nvcc', '-O3', '--compiler-options', args, '-o', out_path, '--shared', source], check=True)
    else:
        raise RuntimeError(f'Unknown file extension {file_ext}')

def do_bench(fn, warmup=25, rep=100, grad_to_none=None,
             quantiles=None,
             fast_flush=True,
             return_mode="mean"):
    assert return_mode in ["min", "max", "mean", "median"]
    import torch
    """
    Benchmark the runtime of the provided function. By default, return the median runtime of :code:`fn` along with
    the 20-th and 80-th performance percentile.

    :param fn: Function to benchmark
    :type fn: Callable
    :param warmup: Warmup time (in ms)
    :type warmup: int
    :param rep: Repetition time (in ms)
    :type rep: int
    :param grad_to_none: Reset the gradient of the provided tensor to None
    :type grad_to_none: torch.tensor, optional
    :param quantiles: Performance percentile to return in addition to the median.
    :type quantiles: list[float]
    :param fast_flush: Use faster kernel to flush L2 between measurements
    :type fast_flush: bool
    """

    fn()
    torch.cuda.synchronize()

    # We maintain a buffer of 256 MB that we clear
    # before each kernel call to make sure that the L2
    # doesn't contain any input data before the run
    if fast_flush:
        cache = torch.empty(int(256e6 // 4), dtype=torch.int, device='cuda')
    else:
        cache = torch.empty(int(256e6), dtype=torch.int8, device='cuda')

    # Estimate the runtime of the function
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    for _ in range(5):
        cache.zero_()
        fn()
    end_event.record()
    torch.cuda.synchronize()
    estimate_ms = start_event.elapsed_time(end_event) / 5

    # compute number of warmup and repeat
    n_warmup = max(1, int(warmup / estimate_ms))
    n_repeat = max(1, int(rep / estimate_ms))
    start_event = [torch.cuda.Event(enable_timing=True) for i in range(n_repeat)]
    end_event = [torch.cuda.Event(enable_timing=True) for i in range(n_repeat)]
    # Warm-up
    for _ in range(n_warmup):
        fn()
    # Benchmark
    for i in range(n_repeat):
        # we don't want `fn` to accumulate gradient values
        # if it contains a backward pass. So we clear the
        # provided gradients
        if grad_to_none is not None:
            for x in grad_to_none:
                x.grad = None
        # we clear the L2 cache before each run
        cache.zero_()
        # record time of `fn`
        start_event[i].record()
        fn()
        end_event[i].record()
    # Record clocks
    torch.cuda.synchronize()
    times = torch.tensor([s.elapsed_time(e) for s, e in zip(start_event, end_event)], dtype=torch.float)
    if quantiles is not None:
        ret = torch.quantile(times, torch.tensor(quantiles, dtype=torch.float)).tolist()
        if len(ret) == 1:
            ret = ret[0]
        return ret
    return getattr(torch, return_mode)(times).item()

def is_fundamental_type(a):
    return isinstance(a, (int, float, str, bool))

CACHE_DIR = '.cache/hip_kernels/'

class KernelConfig:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
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


class KernelHandler:
    def __init__(
            self, 
            source_file: str, 
            compile_configs: List[KernelConfig], 
            keys: List[str], 
            platform: str = 'amd',
            compile_params: Optional[Dict[str, Any]] = None,
            disable_benchmark: bool = False
        ):
        """
        Invariants that must be satisfied by the source file:
        1. The launch function name must be able to be set via LAUNCH_NAME macro passed to the compiler
            ex. -D LAUNCH_NAME=launch_10
        2. The launch function must return a bool indicating whether the kernel launch succeeded
            True: kernel launch succeeded
            False: kernel launch failed
        """ 
        assert platform in ['amd', 'nvidia']
        self.disable_benchmark = disable_benchmark
        self.platform = platform
        # CACHE_DIR
        # |_ hash(source_file)  -> self.dir_path
        # |__ fmt(config[0]).so -> config[0].so_name()
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
        dir_path = os.path.join(CACHE_DIR, file_name_no_ext + '_' + folderhash[:16])
        self.dir_path = dir_path

        self.keys = keys
        self.keys.sort()

        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

        self.so_name2_config: Dict[str, KernelConfig] = {
            config.so_name(): config for config in compile_configs
        }
        need_to_compile = list(filter(
            lambda x: not os.path.exists(os.path.join(dir_path, x)), self.so_name2_config.keys()
        ))
        extra_params = {k: str(v) for k, v in compile_params.items()} if compile_params is not None else {}
        if len(need_to_compile) > 0:
            for so_name in tqdm(need_to_compile, desc='Compiling kernels'):
                config = self.so_name2_config[so_name]
                assert len(config.config.keys() & extra_params.keys()) == 0, f'Extra params {extra_params.keys()} overlap with config keys {config.config.keys()}'
                config.config.update(extra_params)
                launch_name = config.launch_name()
                build(self.source_file, os.path.join(dir_path, so_name), amd=platform=='amd', LAUNCH_NAME=launch_name, **config.config)
        
        # so_name -> so_launch_func
        self.launch_funcs = {}
        for so_name, config in self.so_name2_config.items():
            lib = ctypes.cdll.LoadLibrary(os.path.join(dir_path, so_name))
            assert hasattr(lib, config.launch_name()), f'Library {so_name} does not have launch function {config.launch_name()}'
            launch_func = getattr(lib, config.launch_name())
            launch_func.restype = ctypes.c_bool
            self.launch_funcs[so_name] = launch_func
        
        shutil.copy(self.source_file, dir_path)

        # map from runtime-key to best so_name
        self.kernel_map: Dict[str, str] = {}
        if os.path.exists(os.path.join(dir_path, 'meta_data.json')) and len(need_to_compile) == 0:
            with open(os.path.join(dir_path, 'meta_data.json'), 'r') as f:
                self.kernel_map = json.load(f)
    
    def runtime_key(self, runtime_args: Dict[str, Any]) -> str:
        name = ''
        for k in self.keys:
            assert is_fundamental_type(runtime_args[k]), f'Key {k}: {runtime_args[k]} is not a fundamental type'
            name += f'{k}={runtime_args[k]}_'
        return name[:-1]
    
    def dump_meta(self):
        with open(os.path.join(self.dir_path, 'meta_data.json'), 'w') as f:
            json.dump(self.kernel_map, f)
    
    def __call__(self, *args, **kwargs):
        runtime_args = kwargs
        runtime_key: str = self.runtime_key(runtime_args)
        args = list(args) + list(runtime_args.values())
        for i in range(len(args)):
            if isinstance(args[i], torch.Tensor):
                args[i] = ctypes.c_void_p(args[i].data_ptr())
            if isinstance(args[i], float):
                args[i] = ctypes.c_float(args[i])
        
        if self.disable_benchmark:
            func = next(iter(self.launch_funcs.items()))[1]
            if not func(*args):
                raise RuntimeError(f'Kernel launch failed')
            return
        if runtime_key in self.kernel_map:
            func = self.launch_funcs[self.kernel_map[runtime_key]]
            if not func(*args):
                raise RuntimeError(f'Kernel launch {self.kernel_map[runtime_key]} failed')
        else:
            # benchmark
            best_so_name = None
            best_time = float('inf')
            for so_name, func in self.launch_funcs.items():
                if not func(*args):
                    continue
                time = do_bench(lambda: func(*args))
                if time < best_time:
                    best_time = time
                    best_so_name = so_name
            if best_so_name is None:
                raise RuntimeError('All kernels failed')
            self.kernel_map[runtime_key] = best_so_name
            func(*args)
            self.dump_meta()
    

# build('src/saxpy.cu', 'src/saxpy.so', amd=False, BLOCKSIZE=512, REPEATS=4, LAUNCH_NAME='launch10')
