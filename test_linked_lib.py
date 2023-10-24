# %%
import ctypes

test = ctypes.CDLL('build/libhello.so')

test.do_something()
