# %%
from typing import Optional, Dict, Tuple
import ctypes
import torch
from kernels.utils import do_bench

ck_lib = ctypes.cdll.LoadLibrary('build/libck_gemm.so')
ck_dl_lib = ctypes.cdll.LoadLibrary('build/libck_gemm_dl.so')

launch_fn = ck_lib.ck_gemm_f32
launch_fn.argtypes = (ctypes.c_void_p,ctypes.c_void_p,ctypes.c_void_p,ctypes.c_int,ctypes.c_int,ctypes.c_int,ctypes.c_int)
launch_fn.restype = ctypes.c_bool

launch_fn_dl = ck_dl_lib.ck_gemm_f32
launch_fn_dl.argtypes = (ctypes.c_void_p,ctypes.c_void_p,ctypes.c_void_p,ctypes.c_int,ctypes.c_int,ctypes.c_int)
launch_fn_dl.restype = ctypes.c_bool

ver_name_fn = ck_lib.version_name
ver_name_fn.argtypes = (ctypes.c_int,)
ver_name_fn.restype = ctypes.c_char_p

num_ver_fn = ck_lib.num_versions
num_ver_fn.restype = ctypes.c_int

launch_table: Dict[Tuple[int, int, int], int] = {}

def ck_gemm_dl(a: torch.Tensor, b: torch.Tensor):
    assert len(a.shape) == len(b.shape) == 2
    m, k = a.shape
    _, n = b.shape
    assert tuple(a.shape) == (m, k) and tuple(b.shape) == (k, n)
    c = torch.empty([m, n], dtype=a.dtype, device=a.device)
    success = launch_fn_dl(
        ctypes.c_void_p(a.data_ptr()),
        ctypes.c_void_p(b.data_ptr()),
        ctypes.c_void_p(c.data_ptr()),
        m, k, n
    )
    if not success:
        raise RuntimeError(f'launch failed')
    return c


def ck_gemm(a: torch.Tensor, b: torch.Tensor, ver: Optional[int] = None):
    assert len(a.shape) == len(b.shape) == 2
    m, k = a.shape
    _, n = b.shape
    assert tuple(a.shape) == (m, k) and tuple(b.shape) == (k, n)
    assert isinstance(ver, (int, type(None)))

    if ver is None: # benchmark to find best
        inshape = (m, k, n)
        if inshape in launch_table:
            return ck_gemm(a, b, launch_table[inshape])
        else:
            times = [(i, do_bench(lambda: ck_gemm(a, b, i), return_mode='median')) for i in range(0, ck_num_versions())]
            times.sort(key=lambda x: x[1])
            idx = times[0][0]
            launch_table[inshape] = idx
            return ck_gemm(a, b, idx)
    
    c = torch.empty([m, n], dtype=a.dtype, device=a.device)
    success = launch_fn(
        ctypes.c_void_p(a.data_ptr()),
        ctypes.c_void_p(b.data_ptr()),
        ctypes.c_void_p(c.data_ptr()),
        m, k, n, ver
    )
    if not success:
        raise RuntimeError(f'launch for ver {ver} failed')
    return c

def ck_num_versions() -> int:
    return num_ver_fn()

def ck_ver_name(ver: int) -> str:
    _res = ver_name_fn(ver)
    return _res


if __name__ == '__main__':
    a = torch.randn([512, 512], device='cuda')
    b = torch.randn_like(a)
    c1 = ck_gemm(a, b)
    c2 = a @ b
    err = (c1 - c2).abs()
    print(err.max())
    c3 = ck_gemm_dl(a, b)
    err = (c2 - c3).abs()
    print(err.max())

    print("num kernels:", ck_num_versions())
    for i in range(ck_num_versions()):
        med = do_bench(lambda: ck_gemm(a, b, ver=i), return_mode='median')
        print(ck_ver_name(i), f': {med}s')

