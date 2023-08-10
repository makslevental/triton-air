import re

import mlir_utils.types as T
from triton_mlir_bindings.ir import Type, Value, ShapedType


def ptr(type_or_val: Type | Value):
    if isinstance(type_or_val, Value):
        type_ = type_or_val.type
    else:
        type_ = type_or_val
    if isinstance(type_, Type):
        if ShapedType.isinstance(type_):
            return Type.parse(f"!tt.ptr<{type_.element_type}>")
        else:
            return Type.parse(f"!tt.ptr<{type_}>")


class Ptr(Type):
    def __pos__(self):
        return Ptr(ptr(self))


_p_i16 = lambda: ptr(T.i16)
_p_i32 = lambda: ptr(T.i32)
_p_i64 = lambda: ptr(T.i64)

_p_f16 = lambda: ptr(T.f16)
_p_f32 = lambda: ptr(T.f32)
_p_f64 = lambda: ptr(T.f64)
_p_bf16 = lambda: ptr(T.bf16)

# matches python/triton/language/core.py
_void = lambda: T.none
_int1 = lambda: T.bool
# note that triton thinks these are signed but they're actually signless
_int8 = lambda: T.i8

# _int16 = lambda: T.i16
# _int32 = lambda: T.i32
# _int64 = lambda: T.i64

_int16 = lambda: Ptr(T.i16)
_int32 = lambda: Ptr(T.i32)
_int64 = lambda: Ptr(T.i64)

# triton maps both ui and i to i
_uint8 = lambda: T.i8
_uint16 = lambda: T.i16
_uint32 = lambda: T.i32
_uint64 = lambda: T.i64
_float8e5 = lambda: T.f8e5m2
_float8e4 = lambda: T.f8e4m3
_float8e4b15 = lambda: T.f8e4m3b11fnuz

# _float16 = lambda: T.f16
# _float32 = lambda: T.f32
# _float64 = lambda: T.f64

_float16 = lambda: Ptr(T.f16)
_float32 = lambda: Ptr(T.f32)
_float64 = lambda: Ptr(T.f64)

_bfloat16 = lambda: T.bf16
# pointer types
_pi32 = lambda: ptr(T.i32)

_name_to_type = {
    "p_i16": _p_i16,
    "p_i32": _p_i32,
    "p_i64": _p_i64,
    "p_f16": _p_f16,
    "p_f32": _p_f32,
    "p_f64": _p_f64,
    "p_bf16": _p_bf16,
    "void": _void,
    "int1": _int1,
    "int8": _int8,
    "int16": _int16,
    "int32": _int32,
    "int64": _int64,
    "uint8": _uint8,
    "uint16": _uint16,
    "uint32": _uint32,
    "uint64": _uint64,
    "float8e5": _float8e5,
    "float8e4": _float8e4,
    "float8e4b15": _float8e4b15,
    "float16": _float16,
    "bfloat16": _bfloat16,
    "float32": _float32,
    "float64": _float64,
    "pi32": _pi32,
}


def __getattr__(name):
    if name in _name_to_type:
        return _name_to_type[name]()
    # this kicks it to the default module attribute lookup (i.e., functions defined below and such)
    return None


def is_ptr(o: Type | Value):
    from triton_pp.dialects.ext.triton import TritonPointer

    if isinstance(o, TritonPointer):
        return True
    if not isinstance(o, (Type, Value)):
        return False
    if isinstance(o, Value):
        o = o.type
    if ShapedType.isinstance(o):
        o = ShapedType(o).element_type
    for p in [_p_f16(), _p_f32(), _p_f64(), _p_bf16()]:
        if p.typeid == o.typeid:
            return True
    return False


def get_ptr_type(ptr: Type):
    assert isinstance(ptr, Type), f"{ptr=} is not an mlir type"
    assert "!tt.ptr" in str(ptr), f"{ptr=} is not a tt.ptr"
    ptr_type = re.findall(r"!tt\.ptr<(\w+)>", str(ptr))
    assert len(ptr_type) == 1, f"couldn't find element in {ptr_type=}"
    return Type.parse(ptr_type[0])
