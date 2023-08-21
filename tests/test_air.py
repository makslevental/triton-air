from textwrap import dedent

import pytest
from mlir_utils.dialects.ext.arith import constant
from mlir_utils.dialects.ext.func import func
from mlir_utils.testing import filecheck

import triton_pp.types as T
from triton_pp.dialects import air

# noinspection PyUnresolvedReferences
from triton_pp.util import mlir_ctx_fix as ctx

pytest.mark.usefixtures("ctx")

try:
    from air.mlir.ir import (
        Context,
        Location,
        Module,
        IntegerType,
        InsertionPoint,
        FunctionType,
        IndexType,
        IntegerAttr,
    )
    from air.dialects import air as airdialect
    from air.mlir.dialects import arith
    from air.mlir.passmanager import PassManager

    AIR_INSTALLED = True
except:
    AIR_INSTALLED = False


def make_triton_mod():
    air.channel("channel_0")

    @func
    def forward(arg0: T.T.memref(16, 16, T.int32), arg1: T.T.memref(16, 16, T.int32)):
        c0 = constant(0, index=True)
        c1 = constant(1, index=True)
        c8 = constant(8, index=True)
        c16 = constant(16, index=True)
        air.channel_put(None, [], "channel_0", [], arg0, (c0, c0), (c8, c8), (c16, c1))
        air.channel_get(None, [], "channel_0", [], arg1, (c8, c8), (c8, c8), (c16, c1))

    forward.emit()


def test_smoke(ctx):
    make_triton_mod()
    correct = dedent(
        """\
    module {
      "air.channel"() {sym_name = "channel_0"} : () -> ()
      func.func @forward(%arg0: memref<16x16xi32>, %arg1: memref<16x16xi32>) {
        %c0 = arith.constant 0 : index
        %c1 = arith.constant 1 : index
        %c8 = arith.constant 8 : index
        %c16 = arith.constant 16 : index
        "air.channel.put"(%arg0, %c0, %c0, %c8, %c8, %c16, %c1) {chan_name = @channel_0, operand_segment_sizes = array<i32: 0, 0, 1, 2, 2, 2>} : (memref<16x16xi32>, index, index, index, index, index, index) -> ()
        "air.channel.get"(%arg1, %c8, %c8, %c8, %c8, %c16, %c1) {chan_name = @channel_0, operand_segment_sizes = array<i32: 0, 0, 1, 2, 2, 2>} : (memref<16x16xi32>, index, index, index, index, index, index) -> ()
        return
      }
    }
    """
    )
    filecheck(correct, ctx.module)


@pytest.mark.skipif(not AIR_INSTALLED, reason="air not installed")
def test_air_bindings():
    from air.mlir.dialects import func

    def constructAndPrintInFunc(f):
        print("\nTEST:", f.__name__)
        with Context() as ctx, Location.unknown():
            airdialect.register_dialect(ctx)
            module = Module.create()
            with InsertionPoint(module.body):
                ftype = FunctionType.get(
                    [IntegerType.get_signless(32), IntegerType.get_signless(32)], []
                )
                fop = func.FuncOp(f.__name__, ftype)
                bb = fop.add_entry_block()
                with InsertionPoint(bb):
                    f()
                    func.ReturnOp([])
        module.operation.verify()

    @constructAndPrintInFunc
    def launchOp():
        l = airdialect.LaunchOp("pyLaunch")
        with InsertionPoint(l.body):
            idx_ty = IndexType.get()
            arith.ConstantOp(idx_ty, IntegerAttr.get(idx_ty, 1))
            airdialect.LaunchTerminatorOp()


@pytest.mark.skipif(not AIR_INSTALLED, reason="air not installed")
def test_triton_bindings_to_air_bindings(ctx):
    make_triton_mod()
    triton_mod = str(ctx.module)

    from air.mlir.ir import (
        Context,
        Location,
        Module,
    )
    from air.dialects import air as airdialect

    with Context() as ctx, Location.unknown():
        airdialect.register_dialect(ctx)
        module = Module.parse(triton_mod)

    correct = dedent(
        """\
    module {
      air.channel @channel_0 []
      func.func @forward(%arg0: memref<16x16xi32>, %arg1: memref<16x16xi32>) {
        %c0 = arith.constant 0 : index
        %c1 = arith.constant 1 : index
        %c8 = arith.constant 8 : index
        %c16 = arith.constant 16 : index
        air.channel.put  @channel_0[] (%arg0[%c0, %c0] [%c8, %c8] [%c16, %c1]) : (memref<16x16xi32>)
        air.channel.get  @channel_0[] (%arg1[%c8, %c8] [%c8, %c8] [%c16, %c1]) : (memref<16x16xi32>)
        return
      }
    }
    """
    )
    filecheck(correct, str(module))


@pytest.mark.skipif(not AIR_INSTALLED, reason="air not installed")
def test_triton_bindings_to_air_bindings_pass(ctx):
    make_triton_mod()
    triton_mod = str(ctx.module)

    from air.mlir.ir import (
        Context,
        Location,
        Module,
    )

    with Context() as ctx, Location.unknown():
        airdialect.register_dialect(ctx)
        module = Module.parse(triton_mod)

        pm = PassManager.parse(
            "builtin.module(buffer-results-to-out-params,air-to-async)"
        )
        pm.run(module.operation)

    correct = dedent(
        """\
    module {
      memref.global "private" @channel_0 : memref<i64> = dense<0>
      func.func @forward(%arg0: memref<16x16xi32>, %arg1: memref<16x16xi32>) attributes {llvm.emit_c_interface} {
        %c0 = arith.constant 0 : index
        %c1 = arith.constant 1 : index
        %c8 = arith.constant 8 : index
        %c16 = arith.constant 16 : index
        %0 = memref.get_global @channel_0 : memref<i64>
        %1 = builtin.unrealized_conversion_cast %0 : memref<i64> to memref<i64>
        %2 = builtin.unrealized_conversion_cast %arg0 : memref<16x16xi32> to memref<?x?xi32>
        call @air_channel_put_M0I64_M0D2I32_I64_I64_I64_I64_I64_I64(%1, %2, %c0, %c0, %c8, %c8, %c16, %c1) : (memref<i64>, memref<?x?xi32>, index, index, index, index, index, index) -> ()
        %3 = memref.get_global @channel_0 : memref<i64>
        %4 = builtin.unrealized_conversion_cast %3 : memref<i64> to memref<i64>
        %5 = builtin.unrealized_conversion_cast %arg1 : memref<16x16xi32> to memref<?x?xi32>
        call @air_channel_get_M0I64_M0D2I32_I64_I64_I64_I64_I64_I64(%4, %5, %c8, %c8, %c8, %c8, %c16, %c1) : (memref<i64>, memref<?x?xi32>, index, index, index, index, index, index) -> ()
        return
      }
      func.func private @air_channel_put_M0I64_M0D2I32_I64_I64_I64_I64_I64_I64(memref<i64>, memref<?x?xi32>, index, index, index, index, index, index) attributes {llvm.emit_c_interface}
      func.func private @air_channel_get_M0I64_M0D2I32_I64_I64_I64_I64_I64_I64(memref<i64>, memref<?x?xi32>, index, index, index, index, index, index) attributes {llvm.emit_c_interface}
    }
    """
    )
    filecheck(correct, str(module))
