import re

from mlir_utils.types import (
    f16_t,
    f32_t,
    f64_t,
    bf16_t,
    f8e5m2_t,
    f8e4m3_t,
    f8e4m3b11fnuz_t,
    none_t,
    i8_t,
    i32_t,
    i16_t,
    i64_t,
    bool_t,
)
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


p_f16_t = ptr_t(f16_t)
p_f32_t = ptr_t(f32_t)
p_f64_t = ptr_t(f64_t)
p_bf16_t = ptr_t(bf16_t)

# matches python/triton/language/core.py
void = none_t
int1 = bool_t
# note that triton thinks these are signed but they're actually signless
int8 = i8_t
int16 = i16_t
int32 = i32_t
int64 = i64_t
# triton maps both ui and i to i
uint8 = i8_t
uint16 = i16_t
uint32 = i32_t
uint64 = i64_t
float8e5 = f8e5m2_t
float8e4 = f8e4m3_t
float8e4b15 = f8e4m3b11fnuz_t
float16 = f16_t
bfloat16 = bf16_t
float32 = f32_t
float64 = f64_t
# pointer types
pi32_t = ptr_t(i32_t)


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
    for p in [p_f16_t, p_f32_t, p_f64_t, p_bf16_t]:
        if p.typeid == o.typeid:
            return True
    return False


def get_ptr_type(ptr: Type):
    assert isinstance(ptr, Type), f"{ptr=} is not an mlir type"
    assert "!tt.ptr" in str(ptr), f"{ptr=} is not a tt.ptr"
    ptr_type = re.findall(r"!tt\.ptr<(\w+)>", str(ptr))
    assert len(ptr_type) == 1, f"couldn't find element in {ptr_type=}"
    return Type.parse(ptr_type[0])
