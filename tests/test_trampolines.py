import pytest
from mlir_utils.dialects import triton as tl
from mlir_utils.dialects.ext import arith

# noinspection PyUnresolvedReferences
from triton_pp.util import mlir_ctx_fix as ctx

from mlir_utils.testing import filecheck, MLIRContext

from triton_pp.dialects import air
from triton_pp.dialects.ext import triton

pytest.mark.usefixtures("ctx")


def test_trampoline_with_triton(ctx: MLIRContext):
    @triton.jit
    def kernel_0123():
        c64 = arith.constant(64)
        v0 = tl.get_program_id(axis="x")
        air.channel("bob")

    kernel_0123.emit()

    correct = """\
    module {
      tt.func @kernel_0123() {
        %c64_i32 = arith.constant 64 : i32
        %0 = tt.get_program_id x : i32
        "air.channel"() {sym_name = "bob"} : () -> ()
        tt.return
      }
    }
    """
    ctx.module.operation.verify()
    filecheck(correct, ctx.module)
