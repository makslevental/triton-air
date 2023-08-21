import contextlib

import pytest
from mlir_utils.context import MLIRContext, mlir_mod_ctx
from triton_mlir_bindings.dialects import triton as triton_dialect
from triton_mlir_bindings.ir import register_attribute_builder, FlatSymbolRefAttr


def mlir_ctx() -> MLIRContext:
    with mlir_mod_ctx(allow_unregistered_dialects=True) as ctx:
        triton_dialect.register_dialect(ctx.context)
        yield ctx


mlir_ctx_fix = pytest.fixture(mlir_ctx)


mlir_ctx_man = contextlib.contextmanager(mlir_ctx)


@register_attribute_builder("FlatSymbolRefAttr")
def _flatSymbolRefAttr(x, context):
    return FlatSymbolRefAttr.get(x, context=context)
