import pytest

# this needs to be below the triton_mlir_bindings
from mlir_utils.dialects.ext.tensor import empty

# noinspection PyUnresolvedReferences
from mlir_utils.testing import filecheck, MLIRContext, mlir_ctx as ctx
from triton_mlir_bindings.dialects import triton as triton_dialect

from triton_air.dialects.ext.triton import register_triton_casters

pytest.mark.usefixtures("ctx")


def test_value_caster(ctx: MLIRContext):
    triton_dialect.register_dialect(ctx.context)
    register_triton_casters()
    from mlir_utils.types import f32_t
    from triton_air.types import p_f32_t

    t = empty((10, 10), f32_t)
    assert repr(t) == "Tensor(%0, tensor<10x10xf32>)"

    t = empty((10, 10), p_f32_t)
    assert repr(t) == "TritonTensor(%1, tensor<10x10x!tt.ptr<f32>>)"
