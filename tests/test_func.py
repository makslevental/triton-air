from textwrap import dedent

import pytest

# noinspection PyUnresolvedReferences
from triton_mlir_bindings.dialects import triton as triton_dialect

from triton_air.dialects.ext import triton
from triton_air.dialects.ext.triton import splat, arange, addptr, load, store
from triton_air.types import p_f32_t

from mlir_utils.dialects import triton as tl

# this needs to be below the triton_mlir_bindings
from mlir_utils.dialects.ext import arith

# noinspection PyUnresolvedReferences
from mlir_utils.testing import filecheck, MLIRContext, mlir_ctx as ctx
from mlir_utils.types import i32_t

pytest.mark.usefixtures("ctx")


def test_vadd(ctx: MLIRContext):
    @triton.jit
    def kernel_0123(arg0: p_f32_t, arg1: p_f32_t, arg2: p_f32_t, arg3: i32_t):
        v0 = tl.get_program_id(axis="x")
        c32 = arith.constant(64, i32_t)
        # doesn't until triton catches up to
        # https://github.com/llvm/llvm-project/commit/bfb1ba752655bf09b35c486f6cc9817dbedfb1bb
        # v1 = v0 * c32
        v1 = arith.muli(v0, c32)
        v2 = arange(0, 64)
        v3 = splat(v1, (64,))
        v4 = arith.addi(v3, v2)
        v5 = splat(arg3, (64,))
        v6 = arith.cmpi("slt", v4, v5)
        v7 = splat(arg0, (64,))
        v8 = addptr(v7, v4)
        v9 = load(v8, v6, cache="none", evict="normal", is_volatile=False)
        v10 = splat(arg1, (64,))
        v11 = addptr(v10, v4)
        v12 = load(v11, v6, cache="none", evict="normal", is_volatile=False)
        v13 = arith.addf(v9, v12)
        v14 = splat(arg2, (64,))
        v15 = addptr(v14, v4)
        store(v15, v13, v6)

    kernel_0123.emit()

    ctx.module.operation.verify()
    correct = dedent(
        """\
    module {
      tt.func @kernel_0123(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>, %arg2: !tt.ptr<f32>, %arg3: i32) {
        %0 = tt.get_program_id x : i32
        %c64_i32 = arith.constant 64 : i32
        %1 = arith.muli %0, %c64_i32 : i32
        %2 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32>
        %3 = tt.splat %1 : (i32) -> tensor<64xi32>
        %4 = arith.addi %3, %2 : tensor<64xi32>
        %5 = tt.splat %arg3 : (i32) -> tensor<64xi32>
        %6 = arith.cmpi slt, %4, %5 : tensor<64xi32>
        %7 = tt.splat %arg0 : (!tt.ptr<f32>) -> tensor<64x!tt.ptr<f32>>
        %8 = tt.addptr %7, %4 : tensor<64x!tt.ptr<f32>>, tensor<64xi32>
        %9 = tt.load %8, %6 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<64xf32>
        %10 = tt.splat %arg1 : (!tt.ptr<f32>) -> tensor<64x!tt.ptr<f32>>
        %11 = tt.addptr %10, %4 : tensor<64x!tt.ptr<f32>>, tensor<64xi32>
        %12 = tt.load %11, %6 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<64xf32>
        %13 = arith.addf %9, %12 : tensor<64xf32>
        %14 = tt.splat %arg2 : (!tt.ptr<f32>) -> tensor<64x!tt.ptr<f32>>
        %15 = tt.addptr %14, %4 : tensor<64x!tt.ptr<f32>>, tensor<64xi32>
        tt.store %15, %13, %6 {cache = 1 : i32, evict = 1 : i32} : tensor<64xf32>
        tt.return
      }
    }
    """
    )
    filecheck(correct, ctx.module)
