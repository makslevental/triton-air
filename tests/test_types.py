from textwrap import dedent

import pytest

# noinspection PyUnresolvedReferences
from triton_mlir_bindings.dialects import triton as triton_dialect

from triton_air.dialects import triton as tl
from triton_air.dialects.ext import triton
from triton_air.dialects.ext.triton import splat, arange, addptr, load, store
from triton_air.types import ptr, get_ptr_type

# this needs to be below the triton_mlir_bindings
from mlir_utils.dialects.ext import arith
from mlir_utils.dialects.ext.arith import constant

# noinspection PyUnresolvedReferences
from mlir_utils.testing import filecheck, mlir_ctx as ctx, MLIRContext
from mlir_utils.types import i32_t, f32_t

pytest.mark.usefixtures("ctx")


def test_ptr_type(ctx: MLIRContext):
    triton_dialect.register_dialect(ctx.context)
    p_f32_t = ptr(f32_t)
    assert f32_t.isinstance(get_ptr_type(p_f32_t))

    BLOCK_SIZE = 64

    @triton.jit
    def add_kernel(
        x_ptr: p_f32_t, y_ptr: p_f32_t, output_ptr: p_f32_t, n_elements: i32_t
    ):
        pid = tl.get_program_id(axis=0)
        # doesn't until triton catches up to
        # https://github.com/llvm/llvm-project/commit/bfb1ba752655bf09b35c486f6cc9817dbedfb1bb
        # block_start = pid * c32
        block_start = arith.muli(pid, constant(BLOCK_SIZE, i32_t))
        block_start = splat(block_start, (BLOCK_SIZE,))

        i2 = arange(0, BLOCK_SIZE)
        offsets = arith.addi(block_start, i2)

        n_elements = splat(n_elements, (BLOCK_SIZE,))
        mask = arith.cmpi("slt", offsets, n_elements)

        x_ptr = splat(x_ptr, (BLOCK_SIZE,))
        x_ptr_plus_offsets = addptr(x_ptr, offsets)
        x = load(x_ptr_plus_offsets, mask)

        y_ptr = splat(y_ptr, (BLOCK_SIZE,))
        y_ptr_plus_offsets = addptr(y_ptr, offsets)
        y = load(y_ptr_plus_offsets, mask)

        output = arith.addf(x, y)
        output_ptr = splat(output_ptr, (BLOCK_SIZE,))
        output_ptr = addptr(output_ptr, offsets)

        store(output_ptr, output, mask)

    add_kernel.emit()

    ctx.module.operation.verify()
    correct = dedent(
        """\
    module {
      tt.func @add_kernel(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>, %arg2: !tt.ptr<f32>, %arg3: i32) {
        %0 = tt.get_program_id {axis = 0 : i32} : i32
        %c64_i32 = arith.constant 64 : i32
        %1 = arith.muli %0, %c64_i32 : i32
        %2 = tt.splat %1 : (i32) -> tensor<64xi32>
        %3 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32>
        %4 = arith.addi %2, %3 : tensor<64xi32>
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
