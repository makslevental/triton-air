import re

from mlir_utils.dialects.ext.tensor import Tensor
from mlir_utils.types import f16_t, f32_t, f64_t, bf16_t
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


def is_ptr_t(t: Type):
    for p in [p_f16_t, p_f32_t, p_f64_t, p_bf16_t]:
        if p.typeid == t.typeid:
            return True
    return False


def get_ptr_type(ptr: Type):
    assert isinstance(ptr, Type), f"{ptr=} is not an mlir type"
    assert "!tt.ptr" in str(ptr), f"{ptr=} is not a tt.ptr"
    ptr_type = re.findall(r"!tt\.ptr<(\w+)>", str(ptr))
    assert len(ptr_type) == 1, f"couldn't find element in {ptr_type=}"
    return Type.parse(ptr_type[0])
