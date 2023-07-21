import pytest
from mlir_utils.dialects.ext.tensor import Tensor

# noinspection PyUnresolvedReferences
from mlir_utils.testing import MLIRContext, mlir_ctx as ctx
from mlir_utils.types import f32_t, tensor_t
from triton_mlir_bindings.ir import Type

from triton_air.types import (
    ptr_t,
    get_ptr_type,
    p_f16_t,
    p_f64_t,
    p_bf16_t,
    is_ptr_t,
)

pytest.mark.usefixtures("ctx")


def test_ptr_type(ctx: MLIRContext):
    p_f32_t = ptr_t(f32_t)
    assert f32_t.isinstance(get_ptr_type(p_f32_t))
    assert (
        p_f16_t.typeid
        == p_f32_t.typeid
        == p_f64_t.typeid
        == p_f32_t.typeid
        == p_bf16_t.typeid
        != f32_t.typeid
    )

    assert is_ptr_t(Type.parse(f"!tt.ptr<f32>"))
    assert is_ptr_t(Type.parse(f"!tt.ptr<bf16>"))

    assert is_ptr_t(p_f16_t)
    assert is_ptr_t(p_f32_t)
    assert is_ptr_t(p_f64_t)
    assert is_ptr_t(p_bf16_t)


def test_tensor_ptrs(ctx: MLIRContext):
    p_f32_t = ptr_t(f32_t)
    t = Tensor.empty((10, 10), f32_t)
    t_ptr_t = ptr_t(t)
    assert is_ptr_t(t_ptr_t)
    assert t_ptr_t.typeid == p_f32_t.typeid

    t_f32_ptr_t = tensor_t(10, 10, t_ptr_t)
    assert repr(t_f32_ptr_t) == "RankedTensorType(tensor<10x10x!tt.ptr<f32>>)"
    tt = Tensor.empty((10, 10), t_ptr_t)
    assert tt.type == t_f32_ptr_t
    assert tt.type.typeid == t_f32_ptr_t.typeid

    assert t.type.typeid == tt.type.typeid
    assert not t.dtype.typeid == tt.dtype.typeid

    ctx.module.operation.verify()
