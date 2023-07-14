import re

from triton_mlir_bindings._mlir_libs._mlir.ir import Type


def ptr(type: Type):
    return Type.parse(f"!tt.ptr<{type}>")


def get_ptr_type(ptr: Type):
    assert isinstance(ptr, Type), f"{ptr=} is not an mlir type"
    assert "!tt.ptr" in str(ptr), f"{ptr=} is not a tt.ptr"
    ptr_type = re.findall(r"!tt\.ptr<(\w+)>", str(ptr))
    assert len(ptr_type) == 1, f"couldn't find element in {ptr_type=}"
    return Type.parse(ptr_type[0])
