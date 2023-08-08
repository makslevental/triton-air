import pytest
from mlir_utils.dialects.ext.tensor import empty

# noinspection PyUnresolvedReferences
from triton_air.util import mlir_ctx_fix as ctx

from mlir_utils.testing import MLIRContext
from mlir_utils.types import tensor_t
from triton_mlir_bindings.ir import Type
from triton_air.types import get_ptr_type, ptr_t, is_ptr_t
import triton_air.types as T

pytest.mark.usefixtures("ctx")


def test_ptr_type(ctx: MLIRContext):
    p_f32_t = ptr_t(T.float32)
    assert T.float32.isinstance(get_ptr_type(p_f32_t))
    assert (
        T.p_f16_t.typeid
        == p_f32_t.typeid
        == T.p_f64_t.typeid
        == p_f32_t.typeid
        == T.p_bf16_t.typeid
        != T.float32.typeid
    )

    assert is_ptr_t(Type.parse(f"!tt.ptr<f32>"))
    assert is_ptr_t(Type.parse(f"!tt.ptr<bf16>"))

    assert is_ptr_t(T.p_f16_t)
    assert is_ptr_t(p_f32_t)
    assert is_ptr_t(T.p_f64_t)
    assert is_ptr_t(T.p_bf16_t)


def test_tensor_ptrs(ctx: MLIRContext):
    p_f32_t = ptr_t(T.float32)
    t = empty((10, 10), T.float32)
    t_ptr_t = ptr_t(t)
    assert is_ptr_t(t_ptr_t)
    assert t_ptr_t.typeid == p_f32_t.typeid

    t_f32_ptr_t = tensor_t(10, 10, t_ptr_t)
    assert repr(t_f32_ptr_t) == "RankedTensorType(tensor<10x10x!tt.ptr<f32>>)"
    tt = empty((10, 10), t_ptr_t)
    assert tt.type == t_f32_ptr_t
    assert tt.type.typeid == t_f32_ptr_t.typeid

    assert t.type.typeid == tt.type.typeid
    assert not t.dtype.typeid == tt.dtype.typeid

    ctx.module.operation.verify()
