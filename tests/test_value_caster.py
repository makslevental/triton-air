import pytest

# this needs to be below the triton_mlir_bindings
from mlir_utils.dialects.ext.tensor import empty

# noinspection PyUnresolvedReferences
from triton_pp.util import mlir_ctx_fix as ctx

from mlir_utils.testing import MLIRContext
import triton_pp.types as T

pytest.mark.usefixtures("ctx")


def test_value_caster(ctx: MLIRContext):
    t = empty((10, 10), T.float32)
    assert repr(t) == "Tensor(%0, tensor<10x10xf32>)"

    t = empty((10, 10), +T.float32)
    assert repr(t) == "TritonTensor(%1, tensor<10x10x!tt.ptr<f32>>)"
