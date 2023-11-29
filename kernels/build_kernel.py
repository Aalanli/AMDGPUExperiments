# %%
import copy
from dataclasses import dataclass
import multiprocessing
from typing import List, Dict, Tuple, Any, Optional, Sequence, Iterable, Callable, Set
from tqdm import tqdm
import hashlib
import os
import json
import shutil
import subprocess
import ctypes
import torch

def format_error(result):
    if result.returncode:
        message = ""
        if result.stdout:
            message += result.stdout.decode().strip() + '\n'
        if result.stderr:
            message += result.stderr.decode().strip()
        return message
    else:
        return None

AMDGPU_ARCHS = ('gfx90a', 'gfx1100')

def build(ignore_error, source, out_path, amd=True, archs=('gfx90a',), **kwargs):
    # print(f'Building {source} to {out_path}')
    args = [f'-D {k}={v}' for k, v in kwargs.items()] + ['-fPIC', '-funroll-loops', '-ffast-math', '-O3', '-g', '-std=c++17']
    assert os.path.exists(source) and os.path.isfile(source)

    file = os.path.basename(source)
    file_ext = os.path.splitext(file)[1]
    std_err = subprocess.DEVNULL if ignore_error else None
    if amd:
        assert file_ext == '.cpp', f'AMD kernel must be a cpp file, got {file_ext}'
        for arch in archs:
            assert arch in AMDGPU_ARCHS
            args.append(f'--offload-arch={arch}')
        result = subprocess.run(['hipcc', '-shared', source] + args + ['-o', out_path, '-I', 'include/'], 
                       check=False, shell=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        msg = format_error(result)
        if msg is not None:
            raise Exception(msg)                
        return
    
    args = ' '.join(args[:-1])    
    if file_ext == '.cpp':
        env = os.environ.copy()
        env['HIP_PLATFORM'] = 'nvidia'
        subprocess.run(['hipcc', '-O3', '-c', source, '--compiler-options', args, '-o', out_path, '-I', 'include/'], check=True, env=env, shell=False, stdout=subprocess.DEVNULL, stderr=std_err)
        subprocess.run(['hipcc', '-shared', '-o', out_path, out_path], check=True, shell=False)
    elif file_ext == '.cu':
        result = subprocess.run([
            'nvcc', '-O3', '--compiler-options', args + ' -m64', '-ftz=true', '-prec-div=false', 
            '-lineinfo', '-I', 'include', '-o', out_path, '--shared', source], 
        check=False, shell=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        msg = format_error(result)
        if msg is not None:
            raise Exception(msg)
        
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

@dataclass
class BuildConfig:
    source_file: str
    amd: bool
    compile_params: Dict[str, Any]
    out_path: str
    archs: Tuple[str]

def build_worker(config: BuildConfig, ignore_errors=False):
    if not ignore_errors:
        build(False, config.source_file, config.out_path, amd=config.amd, archs=config.archs, **config.compile_params)
        return True, ''
    try:
        build(True, config.source_file, config.out_path, amd=config.amd, archs=config.archs, **config.compile_params)
    except Exception as e:
        print(f'Failed to compile {config.compile_params}: \ndue to {e}')      
        return False, str(e)
    return True, ''


class JobQueue:
    def __init__(self, func, jobs: Sequence[Any] = tuple()):
        self.func: Callable = func
        self.jobs: Sequence[Any] = jobs


_job_queue: Optional[JobQueue] = None


def _wrapped_func(job_index):
    """
    Wrapper function for parallel_imap.

    We use this function to avoid pickling the jobs.
    """
    assert job_index < len(_job_queue.jobs)

    job = _job_queue.jobs[job_index]
    func = _job_queue.func

    return func(job)


def parallel_imap(func: Callable, jobs: Sequence[Any], num_workers: Optional[int] = None) -> Iterable[Any]:
    global _job_queue

    if _job_queue is not None:
        raise RuntimeError('Cannot call parallel_map recursively.')

    _job_queue = JobQueue(func, jobs)

    if num_workers is None:
        num_workers = os.cpu_count()

    with multiprocessing.Pool(num_workers) as pool:
        yield from pool.imap(_wrapped_func, range(len(jobs)))

    _job_queue = None

class KernelHandler:
    def __init__(
            self, 
            source_file: str, 
            compile_configs: List[KernelConfig], 
            keys: List[str], 
            platform: str = 'amd',
            archs: Tuple[str] = ('gfx90a',),
            warp_size: int = 64,
            name: Optional[str] = None,
            keep: int = 3,
            compile_params: Optional[Dict[str, Any]] = None,
            disable_benchmark: bool = False,
            parallel_compile: bool = True,
            ignore_compile_errors: bool = True
        ):
        """
        Invariants that must be satisfied by the source file:
        1. The launch function name must be able to be set via LAUNCH_NAME macro passed to the compiler
            ex. -D LAUNCH_NAME=launch_10
        2. The launch function must return a bool indicating whether the kernel launch succeeded
            True: kernel launch succeeded
            False: kernel launch failed
        
        Cache Directory Structure:
        CACHE_DIR
          NAME
            1
              kernels (a folder containing .so files)
              meta_data.json (a file containing the best kernel names)
              kernel_times.json (a file containing kernel timings on a runtime shape)
              src_file.cpp (the source file)
        """

        assert platform in ['amd', 'nvidia']
        for arch in archs:
            assert arch in AMDGPU_ARCHS
        self.disable_benchmark = disable_benchmark
        self.platform = platform

        self.source_file = source_file
        assert os.path.exists(self.source_file) and os.path.isfile(self.source_file), f'File {self.source_file} does not exist'
        file_name = os.path.basename(self.source_file)
        file_name_no_ext = os.path.splitext(file_name)[0]
        dir_name = file_name_no_ext if name is None else name

        with open(self.source_file, 'r') as f:
            self.source = f.read()
        
        super_dir_path = os.path.join(CACHE_DIR, dir_name)
        if not os.path.exists(super_dir_path):
            os.makedirs(super_dir_path)

        # scan through the cached dirs to see if one matches the current src file
        sub_versions = [int(s) for s in os.listdir(super_dir_path)]
        sub_versions.sort()
        dir_path = None
        for i in sub_versions:
            src_file = os.path.join(super_dir_path, str(i), file_name)
            if os.path.exists(src_file) and os.path.isfile(src_file):
                with open(src_file, 'r') as f:
                    new_src = f.read()
                    if new_src == self.source:
                        dir_path = os.path.join(super_dir_path, str(i))
                        break
        if dir_path is None:
            dir_path = os.path.join(super_dir_path, str(1 if len(sub_versions) == 0 else sub_versions[-1] + 1))

        # remove past versions
        if len(sub_versions) > keep and keep > 1:
            rm_path = os.path.join(super_dir_path, str(sub_versions[0]))
            if rm_path != dir_path:
                shutil.rmtree(rm_path)

        kernels_path = os.path.join(dir_path, "kernels")
        self.dir_path = dir_path

        self.keys = keys
        self.keys.sort()

        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        if not os.path.exists(kernels_path):
            os.makedirs(kernels_path)
        shutil.copy(self.source_file, dir_path)
        self.compile_file_src = os.path.join(dir_path, file_name)

        self.so_name2_config: Dict[str, KernelConfig] = {
            config.so_name(): config for config in compile_configs
        }
        self.failed_to_compile: Set[str] = set()
        # reset failed json if parallel is false or ignore errors is false
        if os.path.exists(os.path.join(dir_path, 'failed.json')) and parallel_compile and ignore_compile_errors:
            with open(os.path.join(dir_path, 'failed.json'), 'r') as f:
                self.failed_to_compile: Set[str] = set(json.load(f))
        need_to_compile = list(filter(
            lambda x: not os.path.exists(os.path.join(kernels_path, x)) and x not in self.failed_to_compile, self.so_name2_config.keys()
        ))
        extra_params = {k: str(v) for k, v in compile_params.items()} if compile_params is not None else {}
        failed_reasons = ''

        if len(need_to_compile) > 0:
            compile_items: List[BuildConfig] = []
            for so_name in need_to_compile:
                config = self.so_name2_config[so_name]
                assert len(config.config.keys() & extra_params.keys()) == 0, f'Extra params {extra_params.keys()} overlap with config keys {config.config.keys()}'
                config.config.update(extra_params)
                launch_name = config.launch_name()
                compile_param = {}
                compile_param.update(config.config)
                compile_param['LAUNCH_NAME'] = launch_name
                compile_param['warp_size'] = warp_size

                compile_items.append(BuildConfig(
                    source_file=self.source_file,
                    amd=platform=='amd',
                    compile_params=compile_param,
                    out_path=os.path.join(kernels_path, so_name),
                    archs=archs
                ))
            
            if parallel_compile:
                res: List[Tuple[bool, str]] = list(tqdm(parallel_imap(lambda x: build_worker(x, ignore_compile_errors), compile_items), total=len(need_to_compile), desc='Compiling kernels'))
            else:
                res: List[Tuple[bool, str]] = [build_worker(config, ignore_compile_errors) for config in compile_items]
            
            successfully_compiled = 0
            for i in range(len(res)):
                if not res[i][0]: # failed to compile
                    self.failed_to_compile.add(need_to_compile[i])     
                    failed_reasons += need_to_compile[i] + '\n' + res[i][1] + '\n'

                if res[i][0] and need_to_compile[i] in self.failed_to_compile:
                    self.failed_to_compile.remove(need_to_compile[i])
                    successfully_compiled += 1

        if ignore_compile_errors:
            with open(os.path.join(dir_path, 'failed.json'), 'w') as f:
                json.dump(list(self.failed_to_compile), f)
            with open(os.path.join(dir_path, 'failed_reason.txt'), 'w') as f:
                f.write(failed_reasons)
            

        # so_name -> so_launch_func
        self.launch_funcs = {}
        for so_name, config in self.so_name2_config.items():
            lib_path = os.path.join(kernels_path, so_name)
            if not os.path.exists(lib_path) or so_name in self.failed_to_compile:
                continue
            lib = ctypes.cdll.LoadLibrary(os.path.join(kernels_path, so_name))
            assert hasattr(lib, config.launch_name()), f'Library {so_name} does not have launch function {config.launch_name()}'
            launch_func = getattr(lib, config.launch_name())
            launch_func.restype = ctypes.c_bool
            self.launch_funcs[so_name] = launch_func
        if len(self.launch_funcs) == 0:
            raise RuntimeError(f'No kernels compiled for {self.source_file}')

        # map from runtime-key to best so_name
        self.kernel_map: Dict[str, str] = {}
        self.kernel_times: Dict[str, Dict[str, float]] = {}
        if os.path.exists(os.path.join(dir_path, 'meta_data.json')) and len(need_to_compile) == 0:
            with open(os.path.join(dir_path, 'meta_data.json'), 'r') as f:
                self.kernel_map = json.load(f)
        if os.path.exists(os.path.join(dir_path, 'kernel_times.json')):
            with open(os.path.join(dir_path, 'kernel_times.json'), 'r') as f:
                self.kernel_times = json.load(f)
    
    def kernel_avg_time(self) -> List[Tuple[str, float]]:
        kernel_times = {}
        for _runtime_key, variants in self.kernel_times.items():
            for so_name, time in variants.items():
                if so_name in kernel_times:
                    kernel_times[so_name].append(time)
                else:
                    kernel_times[so_name] = [time]
        kernel_avg_time = [(k, sum(v) / len(v)) for k, v in kernel_times.items()]
        kernel_avg_time.sort(key=lambda x: x[1])
        return kernel_avg_time

    def filtered_kernel_times(self, top_k: int) -> Dict[str, Dict[str, float]]:
        kernel_avg_time = self.kernel_avg_time()
        filtered_kernel_times = copy.deepcopy(self.kernel_times)
        for bad_kernel, _ in kernel_avg_time[top_k:]:
            # declutter kernel times 
            for rtk in filtered_kernel_times:
                for so_name in filtered_kernel_times[rtk]:
                    if so_name == bad_kernel:
                        filtered_kernel_times[rtk].pop(so_name)
                        break
        return filtered_kernel_times
    
    def prune_kernels(self, top_k: int):
        kernel_avg_time = self.kernel_avg_time()
        for bad_kernel, _ in kernel_avg_time[top_k:]:
            self.launch_funcs.pop(bad_kernel)
            if bad_kernel in self.kernel_map.values():
                raise RuntimeError(f"removed top kernel {bad_kernel}, but its the top for a runtime shape")
        
    def runtime_key(self, runtime_args: Dict[str, Any]) -> str:
        name = ''
        for k in self.keys:
            assert is_fundamental_type(runtime_args[k]), f'Key {k}: {runtime_args[k]} is not a fundamental type'
            name += f'{k}={runtime_args[k]}_'
        return name[:-1]
    
    def dump_meta(self):
        with open(os.path.join(self.dir_path, 'meta_data.json'), 'w') as f:
            json.dump(self.kernel_map, f, indent=2)
        def order_timings(kernel_times):
            ordered_timings = []
            for k, v in kernel_times.items():
                v = list(v.items())
                v.sort(key=lambda x: x[1])
                ordered_timings.append((k, v))
            ordered_timings.sort(key=lambda x: x[0])
            ordered_timings = {k: {k1: v1 for k1, v1 in v} for k, v in ordered_timings}
            return ordered_timings
        
        with open(os.path.join(self.dir_path, 'kernel_times.json'), 'w') as f:
            json.dump(order_timings(self.kernel_times), f, indent=2)
        with open(os.path.join(self.dir_path, 'kernel_times_pruned.json'), 'w') as f:            
            json.dump(order_timings(self.filtered_kernel_times(20)), f, indent=2)

    def warp_args(self, *args, **kwargs):
        runtime_args = kwargs
        runtime_key: str = self.runtime_key(runtime_args)
        args = list(args) + list(runtime_args.values())
        for i in range(len(args)):
            if isinstance(args[i], torch.Tensor):
                args[i] = ctypes.c_void_p(args[i].data_ptr())
            if isinstance(args[i], float):
                args[i] = ctypes.c_float(args[i])
        return args, runtime_key
        
    def call_so(self, so_name, *args, **kwargs):
        args, _ = self.warp_args(*args, **kwargs)
        func = self.launch_funcs[so_name]
        if not func(*args):
            raise RuntimeError(f'Kernel launch {so_name} failed')

    def __call__(self, *args, **kwargs):
        args, runtime_key = self.warp_args(*args, **kwargs)

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
            if runtime_key not in self.kernel_times:
                self.kernel_times[runtime_key] = {}
            best_so_name = None
            best_time = float('inf')
            for so_name, func in tqdm(self.launch_funcs.items(), desc='Benchmarking kernels'):
                if so_name in self.kernel_times[runtime_key]:
                    time = self.kernel_times[runtime_key][so_name]
                    if time < best_time:
                        best_time = time
                        best_so_name = so_name
                    continue
                if not func(*args):
                    continue
                time = do_bench(lambda: func(*args))
                self.kernel_times[runtime_key][so_name] = time
                if time < best_time:
                    best_time = time
                    best_so_name = so_name
            if best_so_name is None:
                raise RuntimeError('All kernels failed')
            self.kernel_map[runtime_key] = best_so_name
            func(*args)
            self.dump_meta()
    

# build('src/saxpy.cu', 'src/saxpy.so', amd=False, BLOCKSIZE=512, REPEATS=4, LAUNCH_NAME='launch10')
