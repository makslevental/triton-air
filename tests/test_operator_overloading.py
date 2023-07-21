from textwrap import dedent

import pytest
from mlir_utils.dialects import triton as tl
from mlir_utils.dialects.ext import arith
from mlir_utils.dialects.ext.tensor import empty

# noinspection PyUnresolvedReferences
from mlir_utils.testing import mlir_ctx as ctx, filecheck, MLIRContext
from mlir_utils.types import i32_t

from triton_air.dialects.ext.triton import jit
from triton_air.dialects.ext.triton import splat, arange, load, store
from triton_air.types import p_f32_t

# needed since the fix isn't defined here nor conftest.py
pytest.mark.usefixtures("ctx")


def test_tensor_arithmetic(ctx: MLIRContext):
    # number of elements must be power of 2
    t_p_f32 = empty((16, 16), p_f32_t)
    t_i32 = empty((16, 16), i32_t)
    res = t_p_f32 + t_i32

    ctx.module.operation.verify()
    filecheck(
        dedent(
            """\
    module {
      %0 = tensor.empty() : tensor<16x16x!tt.ptr<f32>>
      %1 = tensor.empty() : tensor<16x16xi32>
      %2 = tt.addptr %0, %1 : tensor<16x16x!tt.ptr<f32>>, tensor<16x16xi32>
    }
    """
        ),
        ctx.module,
    )


def test_value_caster_kernel(ctx: MLIRContext):
    @jit
    def kernel_0123(arg0: p_f32_t, arg1: p_f32_t, arg2: p_f32_t, arg3: i32_t):
        v0 = tl.get_program_id(axis="x")
        c32 = arith.constant(64, i32_t)
        v1 = v0 * c32
        v2 = arange(0, 64)
        v3 = splat(v1, (64,))
        v4 = v3 + v2
        v5 = splat(arg3, (64,))
        v6 = v4 < v5
        v7 = splat(arg0, (64,))
        v8 = v7 + v4
        v9 = load(v8, v6, cache="none", evict="normal", is_volatile=False)
        v10 = splat(arg1, (64,))
        v11 = v10 + v4
        v12 = load(v11, v6, cache="none", evict="normal", is_volatile=False)
        v13 = v9 + v12
        v14 = splat(arg2, (64,))
        v15 = v14 + v4
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
        %6 = arith.cmpi ult, %4, %5 : tensor<64xi32>
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
