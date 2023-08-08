import re

import mlir_utils.types as T
from triton_mlir_bindings.ir import Type, Value, ShapedType


def ptr_t(type_or_val: Type | Value):
    if isinstance(type_or_val, Value):
        type_ = type_or_val.type
    else:
        type_ = type_or_val
    if isinstance(type_, Type):
        if ShapedType.isinstance(type_):
            return Type.parse(f"!tt.ptr<{type_.element_type}>")
        else:
            return Type.parse(f"!tt.ptr<{type_}>")


_p_i16_t = lambda: ptr_t(T.i16_t)
_p_i32_t = lambda: ptr_t(T.i32_t)
_p_i64_t = lambda: ptr_t(T.i64_t)

_p_f16_t = lambda: ptr_t(T.f16_t)
_p_f32_t = lambda: ptr_t(T.f32_t)
_p_f64_t = lambda: ptr_t(T.f64_t)
_p_bf16_t = lambda: ptr_t(T.bf16_t)

# matches python/triton/language/core.py
_void = lambda: T.none_t
_int1 = lambda: T.bool_t
# note that triton thinks these are signed but they're actually signless
_int8 = lambda: T.i8_t
_int16 = lambda: T.i16_t
_int32 = lambda: T.i32_t
_int64 = lambda: T.i64_t
# triton maps both ui and i to i
_uint8 = lambda: T.i8_t
_uint16 = lambda: T.i16_t
_uint32 = lambda: T.i32_t
_uint64 = lambda: T.i64_t
_float8e5 = lambda: T.f8e5m2_t
_float8e4 = lambda: T.f8e4m3_t
_float8e4b15 = lambda: T.f8e4m3b11fnuz_t
_float16 = lambda: T.f16_t
_bfloat16 = lambda: T.bf16_t
_float32 = lambda: T.f32_t
_float64 = lambda: T.f64_t
# pointer types
_pi32_t = lambda: ptr_t(T.i32_t)

_name_to_type = {
    "p_i16_t": _p_i16_t,
    "p_i32_t": _p_i32_t,
    "p_i64_t": _p_i64_t,
    "p_f16_t": _p_f16_t,
    "p_f32_t": _p_f32_t,
    "p_f64_t": _p_f64_t,
    "p_bf16_t": _p_bf16_t,
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
    "pi32_t": _pi32_t,
}


def __getattr__(name):
    if name in _name_to_type:
        return _name_to_type[name]()
    # this kicks it to the default module attribute lookup (i.e., functions defined below and such)
    return None


def is_ptr_t(o: Type | Value):
    from triton_air.dialects.ext.triton import TritonPointer

    if isinstance(o, TritonPointer):
        return True
    if not isinstance(o, (Type, Value)):
        return False
    if isinstance(o, Value):
        o = o.type
    if ShapedType.isinstance(o):
        o = ShapedType(o).element_type
    for p in [_p_f16_t(), _p_f32_t(), _p_f64_t(), _p_bf16_t()]:
        if p.typeid == o.typeid:
            return True
    return False


def get_ptr_type(ptr: Type):
    assert isinstance(ptr, Type), f"{ptr=} is not an mlir type"
    assert "!tt.ptr" in str(ptr), f"{ptr=} is not a tt.ptr"
    ptr_type = re.findall(r"!tt\.ptr<(\w+)>", str(ptr))
    assert len(ptr_type) == 1, f"couldn't find element in {ptr_type=}"
    return Type.parse(ptr_type[0])
