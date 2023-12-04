# %%
from typing import List, Tuple, Dict, Union, Optional
from dataclasses import dataclass


@dataclass(frozen=True)
class Type:
    name: str


@dataclass
class Value:
    name: str
    type: Type


@dataclass
class Op:
    name: str
    args: List[Value]
    ret: List[Value]
    blocks: List['Block']


@dataclass
class Block:
    name: str
    args: List[Value]
    ops: List[Op]

@dataclass(frozen=True)
class Dtype(Type):
    bits: int

i32 = Dtype('i32', 32)
i64 = Dtype('i64', 64)
f32 = Dtype('f32', 32)
f64 = Dtype('f64', 64)

class TensorTrait:
    def dtype(self) -> Dtype:
        raise NotImplementedError()
    
    def shape(self) -> Tuple[Union[Value, int], ...]:
        raise NotImplementedError()

class ShapeOf(Op):
    def __init__(self, tensor: Value, dim: int):
        assert isinstance(tensor.type, TensorTrait)
        super().__init__('shape_of', [tensor], [Value(f'{tensor.name}_shape', Dtype('shape', 32))], [])
        self.dim = dim

class Index(Op):
    def __init__(self, tensor: Value, index: Tuple[Value, ...]):
        assert isinstance(tensor.type, TensorTrait)
        assert len(index) == len(tensor.type.shape)
        super().__init__('index', [tensor, *index], [Value(f'{tensor.name}_index', tensor.type.dtype())], [])
        self.index = index


class Tensor(Type, TensorTrait):
    def __init__(self, dtype: Dtype, shape: Tuple[Union[Value, int], ...]):
        super().__init__(f'tensor')
        self._shape: Tuple[Union[Value, int], ...] = shape
        self._dtype: Dtype = dtype
    
    def __eq__(self, other):
        return isinstance(other, Tensor) and self.dtype == other.dtype and self.shape == other.shape
    
    def __hash__(self):
        return hash((self.dtype, self.shape))
    
    def dtype(self) -> Dtype:
        return self._dtype
    
    def shape(self) -> Tuple[Union[Value, int], ...]:
        return self._shape
    

class TensorSlice(Type, TensorTrait):
    def __init__(self, parent: Tensor, slice: Tuple[slice]):
        super().__init__(f'tensor_slice')
        self.parent: Tensor = parent
        self.slice: Tuple[slice] = slice

        # TODO: check invariants
        self.new_shape: Tuple[Value, ...] = tuple(Value(f'{parent.name}_slice_dim_{i}', i32) for i in range(len(slice)))

    def dtype(self) -> Dtype:
        return self.parent.dtype()
    
    def shape(self) -> Tuple[Union[Value, int], ...]:
        return self.new_shape


class Gemm(Op):
    """
    Semantically: c <- a @ b
    """
    def __init__(self, a: Value, b: Value, c: Value):
        assert isinstance(a.type, TensorTrait) and len(a.type.shape) == 2
        assert isinstance(b.type, TensorTrait) and len(b.type.shape) == 2
        assert isinstance(c.type, TensorTrait) and len(c.type.shape) == 2
        assert a.type.dtype == b.type.dtype == c.type.dtype

        super().__init__('gemm', [a, b, c], [], [])


class Grid(Op):
    """
    Semantically: 
        grid(x, y, z) { inner(x, y, z) }
    """
    def __init__(self, x: Value, y: Value, z: Value, inner: Block):
        assert isinstance(x.type, Dtype) and isinstance(y.type, Dtype) and isinstance(z.type, Dtype)
        assert x.type == y.type and x.type == z.type == i32
        assert isinstance(inner, Block) and len(inner.args) == 3 and all(isinstance(arg.type, Dtype) and arg.type == i32 for arg in inner.args)
        super().__init__('grid', [x, y, z], [], [inner])


class TiledFor(Op):
    pass


