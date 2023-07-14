import pytest
from mlir_utils.dialects.ext import arith

# noinspection PyUnresolvedReferences
from mlir_utils.testing import filecheck, mlir_ctx as ctx, MLIRContext
from mlir_utils.dialects import triton as tl

from triton_air.dialects import air
from triton_air.dialects.ext import triton

pytest.mark.usefixtures("ctx")


def test_trampoline_with_triton(ctx: MLIRContext):
    @triton.jit
    def kernel_0123():
        c64 = arith.constant(64)
        v0 = tl.get_program_id(axis="x")
        air.channel("bob")

    kernel_0123()

    correct = """\
    module {
      tt.func @kernel_0123() {
        %c64_i64 = arith.constant 64 : i64
        %0 = tt.get_program_id x : i32
        "air.channel"() {sym_name = "bob"} : () -> ()
        tt.return
      }
      tt.call @kernel_0123() : () -> ()
    }
    """
    ctx.module.operation.verify()
    filecheck(correct, ctx.module)
