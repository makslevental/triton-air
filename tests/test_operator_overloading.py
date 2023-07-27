from textwrap import dedent

import pytest
from mlir_utils.dialects.ext import arith
from mlir_utils.dialects.ext.tensor import empty

# noinspection PyUnresolvedReferences
from mlir_utils.testing import mlir_ctx as ctx, filecheck, MLIRContext
from mlir_utils.types import i32_t

from triton_air.dialects.ext import triton as tl
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


def test_vadd(ctx: MLIRContext):
    BLOCK_SIZE = 64

    @tl.jit
    def kernel_0123(
        x_ptr: p_f32_t, y_ptr: p_f32_t, output_ptr: p_f32_t, n_elements: i32_t
    ):
        pid = tl.program_id(axis="x")
        block_size = arith.constant(BLOCK_SIZE, i32_t)
        block_start = pid * block_size
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements

        x = tl.load(x_ptr + offsets, mask)
        y = tl.load(y_ptr + offsets, mask)

        output = x + y
        tl.store(output_ptr + offsets, output, mask)

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


def test_vadd_set_get(ctx: MLIRContext):
    BLOCK_SIZE = 64

    @tl.jit
    def kernel_0123(
        x_ptr: p_f32_t, y_ptr: p_f32_t, output_ptr: p_f32_t, n_elements: i32_t
    ):
        pid = tl.program_id(axis="x")
        block_size = arith.constant(BLOCK_SIZE, i32_t)
        block_start = pid * block_size
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements

        x_ptr += offsets
        x = x_ptr[mask]
        y_ptr += offsets
        y = y_ptr[mask]

        output = x + y
        output_ptr += offsets
        output_ptr[mask] = output

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


def test_matmul(ctx: MLIRContext):
    BLOCK_SIZE_M = 64
    BLOCK_SIZE_N = 64
    BLOCK_SIZE_K = 64
    GROUP_SIZE_M = 64

    @tl.jit
    def matmul_kernel(
        a_ptr: p_f32_t,
        b_ptr: p_f32_t,
        c_ptr: p_f32_t,
        # Matrix dimensions
        M: i32_t,
        N: i32_t,
        K: i32_t,
        # The stride variables represent how much to increase the ptr by when moving by 1
        # element in a particular dimension. E.g. `stride_am` is how much to increase `a_ptr`
        # by to get the element one row down (A has M rows).
        stride_am: i32_t,
        stride_ak: i32_t,
        stride_bk: i32_t,
        stride_bn: i32_t,
        stride_cm: i32_t,
        stride_cn: i32_t,
        # Meta-parameters
    ):
        pid = tl.program_id(axis="x")
        num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
        num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
        num_pid_in_group = GROUP_SIZE_M * num_pid_n
        group_id = pid // num_pid_in_group
        first_pid_m = group_id * GROUP_SIZE_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
        pid_m = first_pid_m + (pid % group_size_m)
        pid_n = (pid % num_pid_in_group) // group_size_m

        offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
        offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
        offs_k = tl.arange(0, BLOCK_SIZE_K)
        a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
        b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    matmul_kernel.emit()

    print(ctx.module)
    ctx.module.operation.verify()
