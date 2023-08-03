from textwrap import dedent

import pytest
from mlir_utils.dialects.ext import arith
from mlir_utils.dialects.ext.scf import range_, yield_
from mlir_utils.dialects.ext.tensor import empty

# noinspection PyUnresolvedReferences
from mlir_utils.testing import mlir_ctx as ctx, filecheck, MLIRContext
from mlir_utils.types import i32_t
from triton_mlir_bindings.passmanager import PassManager

from triton_air.dialects.ext import triton as tl
from triton_air.types import p_f32_t, float32

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


def test_matmul(ctx: MLIRContext):
    BLOCK_SIZE_M = 16
    BLOCK_SIZE_N = 16
    BLOCK_SIZE_K = 16
    GROUP_SIZE_M = 2

    @tl.jit
    def matmul_kernel(
        a_ptr: p_f32_t,
        b_ptr: p_f32_t,
        c_ptr: p_f32_t,
        M: i32_t,
        N: i32_t,
        K: i32_t,
        stride_am: i32_t,
        stride_ak: i32_t,
        stride_bk: i32_t,
        stride_bn: i32_t,
        stride_cm: i32_t,
        stride_cn: i32_t,
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

        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=float32)
        for k, (acc, aptrs, bptrs) in range_(
            0, tl.cdiv(K, BLOCK_SIZE_K), iter_args=[accumulator, a_ptrs, b_ptrs]
        ):
            mask = offs_k[None, :] < K - k * BLOCK_SIZE_K
            a = tl.load(a_ptrs, mask=mask, other=0.0)
            mask = offs_k[:, None] < K - k * BLOCK_SIZE_K
            b = tl.load(b_ptrs, mask=mask, other=0.0)
            # TODO(max): the problem here is the _update_frame_vars upstream
            acc_next = acc + tl.dot(a, b)
            aptrs_next = aptrs + BLOCK_SIZE_K * stride_ak
            bptrs_next = bptrs + BLOCK_SIZE_K * stride_bk
            yield_(acc_next, aptrs_next, bptrs_next)

        c = acc

        offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
        c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
        tl.store(c_ptrs, c, mask=c_mask)

    matmul_kernel.emit()

    ctx.module.operation.verify()
    pm = PassManager.parse("builtin.module(cse)")
    pm.run(ctx.module.operation)

    correct = dedent(
        """\
    module {
      tt.func @matmul_kernel(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>, %arg2: !tt.ptr<f32>, %arg3: i32, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32, %arg9: i32, %arg10: i32, %arg11: i32) {
        %0 = tt.get_program_id x : i32
        %c16_i32 = arith.constant 16 : i32
        %1 = arith.divsi %arg3, %c16_i32 : i32
        %2 = arith.divsi %arg4, %c16_i32 : i32
        %c2_i32 = arith.constant 2 : i32
        %3 = arith.muli %2, %c2_i32 : i32
        %4 = arith.floordivsi %0, %3 : i32
        %5 = arith.muli %4, %c2_i32 : i32
        %6 = arith.subi %1, %5 : i32
        %7 = arith.remsi %0, %c2_i32 : i32
        %8 = arith.addi %5, %7 : i32
        %9 = arith.remsi %0, %3 : i32
        %10 = arith.floordivsi %9, %c2_i32 : i32
        %11 = arith.muli %8, %c16_i32 : i32
        %12 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
        %13 = tt.splat %11 : (i32) -> tensor<16xi32>
        %14 = arith.addi %13, %12 : tensor<16xi32>
        %15 = tt.splat %arg3 : (i32) -> tensor<16xi32>
        %16 = arith.remsi %14, %15 : tensor<16xi32>
        %17 = arith.muli %10, %c16_i32 : i32
        %18 = tt.splat %17 : (i32) -> tensor<16xi32>
        %19 = arith.addi %18, %12 : tensor<16xi32>
        %20 = tt.splat %arg4 : (i32) -> tensor<16xi32>
        %21 = arith.remsi %19, %20 : tensor<16xi32>
        %22 = tt.expand_dims %16 {axis = 1 : i32} : (tensor<16xi32>) -> tensor<16x1xi32>
        %23 = tt.splat %arg6 : (i32) -> tensor<16x1xi32>
        %24 = arith.muli %22, %23 : tensor<16x1xi32>
        %25 = tt.expand_dims %12 {axis = 0 : i32} : (tensor<16xi32>) -> tensor<1x16xi32>
        %26 = tt.splat %arg7 : (i32) -> tensor<1x16xi32>
        %27 = arith.muli %25, %26 : tensor<1x16xi32>
        %28 = tt.broadcast %24 : (tensor<16x1xi32>) -> tensor<16x16xi32>
        %29 = tt.broadcast %27 : (tensor<1x16xi32>) -> tensor<16x16xi32>
        %30 = arith.addi %28, %29 : tensor<16x16xi32>
        %31 = tt.splat %arg0 : (!tt.ptr<f32>) -> tensor<16x16x!tt.ptr<f32>>
        %32 = tt.addptr %31, %30 : tensor<16x16x!tt.ptr<f32>>, tensor<16x16xi32>
        %33 = tt.expand_dims %12 {axis = 1 : i32} : (tensor<16xi32>) -> tensor<16x1xi32>
        %34 = tt.splat %arg8 : (i32) -> tensor<16x1xi32>
        %35 = arith.muli %33, %34 : tensor<16x1xi32>
        %36 = tt.expand_dims %21 {axis = 0 : i32} : (tensor<16xi32>) -> tensor<1x16xi32>
        %37 = tt.splat %arg9 : (i32) -> tensor<1x16xi32>
        %38 = arith.muli %36, %37 : tensor<1x16xi32>
        %39 = tt.broadcast %35 : (tensor<16x1xi32>) -> tensor<16x16xi32>
        %40 = tt.broadcast %38 : (tensor<1x16xi32>) -> tensor<16x16xi32>
        %41 = arith.addi %39, %40 : tensor<16x16xi32>
        %42 = tt.splat %arg1 : (!tt.ptr<f32>) -> tensor<16x16x!tt.ptr<f32>>
        %43 = tt.addptr %42, %41 : tensor<16x16x!tt.ptr<f32>>, tensor<16x16xi32>
        %cst = arith.constant dense<1.000000e+00> : tensor<16x16xf32>
        %44 = arith.divsi %arg5, %c16_i32 : i32
        %c0 = arith.constant 0 : index
        %45 = arith.index_cast %44 : i32 to index
        %c1 = arith.constant 1 : index
        %46:3 = scf.for %arg12 = %c0 to %45 step %c1 iter_args(%arg13 = %cst, %arg14 = %32, %arg15 = %43) -> (tensor<16x16xf32>, tensor<16x16x!tt.ptr<f32>>, tensor<16x16x!tt.ptr<f32>>) {
          %c16 = arith.constant 16 : index
          %65 = arith.muli %arg12, %c16 : index
          %66 = arith.index_cast %65 : index to i32
          %67 = arith.subi %arg5, %66 : i32
          %68 = tt.splat %67 : (i32) -> tensor<1x16xi32>
          %69 = arith.cmpi slt, %25, %68 : tensor<1x16xi32>
          %cst_0 = arith.constant 0.000000e+00 : f32
          %70 = tt.splat %cst_0 : (f32) -> tensor<16x16xf32>
          %71 = tt.broadcast %69 : (tensor<1x16xi1>) -> tensor<16x16xi1>
          %72 = tt.load %32, %71, %70 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<16x16xf32>
          %73 = tt.splat %67 : (i32) -> tensor<16x1xi32>
          %74 = arith.cmpi slt, %33, %73 : tensor<16x1xi32>
          %75 = tt.broadcast %74 : (tensor<16x1xi1>) -> tensor<16x16xi1>
          %76 = tt.load %43, %75, %70 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<16x16xf32>
          %cst_1 = arith.constant 1.000000e+00 : f32
          %77 = tt.splat %cst_1 : (f32) -> tensor<16x16xf32>
          %78 = tt.dot %72, %76, %77 {allowTF32 = true} : tensor<16x16xf32> * tensor<16x16xf32> -> tensor<16x16xf32>
          %79 = arith.addf %arg13, %78 : tensor<16x16xf32>
          %80 = arith.muli %arg7, %c16_i32 : i32
          %81 = tt.splat %80 : (i32) -> tensor<16x16xi32>
          %82 = tt.addptr %arg14, %81 : tensor<16x16x!tt.ptr<f32>>, tensor<16x16xi32>
          %83 = arith.muli %arg8, %c16_i32 : i32
          %84 = tt.splat %83 : (i32) -> tensor<16x16xi32>
          %85 = tt.addptr %arg15, %84 : tensor<16x16x!tt.ptr<f32>>, tensor<16x16xi32>
          scf.yield %79, %82, %85 : tensor<16x16xf32>, tensor<16x16x!tt.ptr<f32>>, tensor<16x16x!tt.ptr<f32>>
        }
        %47 = tt.expand_dims %14 {axis = 1 : i32} : (tensor<16xi32>) -> tensor<16x1xi32>
        %48 = tt.splat %arg10 : (i32) -> tensor<16x1xi32>
        %49 = arith.muli %48, %47 : tensor<16x1xi32>
        %50 = tt.splat %arg2 : (!tt.ptr<f32>) -> tensor<16x1x!tt.ptr<f32>>
        %51 = tt.addptr %50, %49 : tensor<16x1x!tt.ptr<f32>>, tensor<16x1xi32>
        %52 = tt.expand_dims %19 {axis = 0 : i32} : (tensor<16xi32>) -> tensor<1x16xi32>
        %53 = tt.splat %arg11 : (i32) -> tensor<1x16xi32>
        %54 = arith.muli %53, %52 : tensor<1x16xi32>
        %55 = tt.broadcast %54 : (tensor<1x16xi32>) -> tensor<16x1xi32>
        %56 = tt.addptr %51, %55 : tensor<16x1x!tt.ptr<f32>>, tensor<16x1xi32>
        %57 = tt.splat %arg3 : (i32) -> tensor<16x1xi32>
        %58 = arith.cmpi slt, %47, %57 : tensor<16x1xi32>
        %59 = tt.splat %arg4 : (i32) -> tensor<1x16xi32>
        %60 = arith.cmpi slt, %52, %59 : tensor<1x16xi32>
        %61 = tt.broadcast %58 : (tensor<16x1xi1>) -> tensor<16x16xi1>
        %62 = tt.broadcast %60 : (tensor<1x16xi1>) -> tensor<16x16xi1>
        %63 = arith.andi %61, %62 : tensor<16x16xi1>
        %64 = tt.broadcast %56 : (tensor<16x1x!tt.ptr<f32>>) -> tensor<16x16x!tt.ptr<f32>>
        tt.store %64, %46#0, %63 {cache = 1 : i32, evict = 1 : i32} : tensor<16x16xf32>
        tt.return
      }
    }
    """
    )

    filecheck(correct, ctx.module)
