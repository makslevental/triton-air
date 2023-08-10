import pytest
from mlir_utils.context import MLIRContext
from mlir_utils.dialects import triton as tt

# noinspection PyUnresolvedReferences
from triton_pp.util import mlir_ctx_fix as ctx

from mlir_utils.testing import filecheck
from triton_mlir_bindings.dialects import (
    arith as arith_dialect,
)
from triton_mlir_bindings.ir import IntegerType

from triton_pp.dialects import air
import triton_pp.types as T

pytest.mark.usefixtures("ctx")


def test_smoke(ctx):
    air.channel("bob")
    c64 = arith_dialect.ConstantOp(T.int32, 64)
    correct = """\
    module {
      "air.channel"() {sym_name = "bob"} : () -> ()
      %c64_i32 = arith.constant 64 : i32
    }
    """
    ctx.module.operation.verify()
    filecheck(correct, ctx.module)


def test_smoke_with_triton(ctx: MLIRContext):
    i32 = IntegerType.get_signless(32)

    @tt.FuncOp.from_py_func(results=[])
    def kernel_0123():
        c64 = arith_dialect.ConstantOp(i32, 64)
        v0 = tt.GetProgramIdOp(axis="x")
        air.ChannelOp("bob")
        tt.ReturnOp([])

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
