import pytest
from mlir_utils.dialects.ext.tensor import empty

# noinspection PyUnresolvedReferences
from triton_pp.util import mlir_ctx_fix as ctx

from mlir_utils.testing import MLIRContext
from mlir_utils.types import tensor
from triton_mlir_bindings.ir import Type
from triton_pp.types import get_ptr_type, ptr, is_ptr
import triton_pp.types as T

pytest.mark.usefixtures("ctx")


def test_ptr_type(ctx: MLIRContext):
    p_f32 = ptr(T.float32)
    assert T.float32.maybe_downcast().isinstance(get_ptr_type(p_f32))
    assert (
        T.p_f16.typeid
        == p_f32.typeid
        == T.p_f64.typeid
        == p_f32.typeid
        == T.p_bf16.typeid
        != T.float32.typeid
    )

    assert is_ptr(Type.parse(f"!tt.ptr<f32>"))
    assert is_ptr(Type.parse(f"!tt.ptr<bf16>"))

    assert is_ptr(T.p_f16)
    assert is_ptr(p_f32)
    assert is_ptr(T.p_f64)
    assert is_ptr(T.p_bf16)


def test_tensor_ptrs(ctx: MLIRContext):
    p_f32 = ptr(T.float32)
    t = empty((10, 10), T.float32)
    t_ptr = ptr(t)
    assert is_ptr(t_ptr)
    assert t_ptr.typeid == p_f32.typeid

    t_f32_ptr = tensor(10, 10, t_ptr)
    assert repr(t_f32_ptr) == "RankedTensorType(tensor<10x10x!tt.ptr<f32>>)"
    tt = empty((10, 10), t_ptr)
    assert tt.type == t_f32_ptr
    assert tt.type.typeid == t_f32_ptr.typeid

    assert t.type.typeid == tt.type.typeid
    assert not t.dtype.typeid == tt.dtype.typeid

    ctx.module.operation.verify()


def test_plus_ptrs(ctx: MLIRContext):
    p_i16 = ptr(T.int16)
    p_i32 = ptr(T.int32)
    p_i64 = ptr(T.int64)

    p_f16 = ptr(T.float16)
    p_f32 = ptr(T.float32)
    p_f64 = ptr(T.float64)

    pp_i16 = +T.int16
    pp_i32 = +T.int32
    pp_i64 = +T.int64

    assert pp_i16 == p_i16
    assert pp_i32 == p_i32
    assert pp_i64 == p_i64

    pp_f16 = +T.float16
    pp_f32 = +T.float32
    pp_f64 = +T.float64

    assert pp_f16 == p_f16
    assert pp_f32 == p_f32
    assert pp_f64 == p_f64

    x = +++++++++T.int64
    assert str(x) == "!tt.ptr<!tt.ptr<!tt.ptr<!tt.ptr<!tt.ptr<!tt.ptr<!tt.ptr<!tt.ptr<!tt.ptr<i64>>>>>>>>>"