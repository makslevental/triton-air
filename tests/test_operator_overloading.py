from textwrap import dedent

import pytest
from mlir_utils.dialects.ext import arith
from mlir_utils.dialects.ext.scf import range_, yield_
from mlir_utils.dialects.ext.tensor import empty

# noinspection PyUnresolvedReferences
from mlir_utils.testing import mlir_ctx as ctx, filecheck, MLIRContext
from triton_mlir_bindings.dialects import triton as triton_dialect
from triton_mlir_bindings.passmanager import PassManager

from triton_air.dialects.ext import triton as tl
from triton_air.dialects.ext.triton import register_triton_casters

# needed since the fix isn't defined here nor conftest.py
pytest.mark.usefixtures("ctx")


def test_tensor_arithmetic(ctx: MLIRContext):
    triton_dialect.register_dialect(ctx.context)
    register_triton_casters()
    from triton_air.types import p_f32_t
    from mlir_utils.types import i32_t

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
    triton_dialect.register_dialect(ctx.context)
    from triton_air.types import p_f32_t
    from mlir_utils.types import i32_t

    register_triton_casters()

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
    triton_dialect.register_dialect(ctx.context)
    register_triton_casters()
    from triton_air.types import p_f32_t
    from mlir_utils.types import i32_t

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
    triton_dialect.register_dialect(ctx.context)
    register_triton_casters()
    from triton_air.types import p_f32_t, float32
    from mlir_utils.types import i32_t

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

        # offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
        # offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
        offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
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
            acc_res, *_ = yield_(acc_next, aptrs_next, bptrs_next)

        c = acc_res

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
        %15 = arith.muli %10, %c16_i32 : i32
        %16 = tt.splat %15 : (i32) -> tensor<16xi32>
        %17 = arith.addi %16, %12 : tensor<16xi32>
        %18 = tt.expand_dims %14 {axis = 1 : i32} : (tensor<16xi32>) -> tensor<16x1xi32>
        %19 = tt.splat %arg6 : (i32) -> tensor<16x1xi32>
        %20 = arith.muli %18, %19 : tensor<16x1xi32>
        %21 = tt.expand_dims %12 {axis = 0 : i32} : (tensor<16xi32>) -> tensor<1x16xi32>
        %22 = tt.splat %arg7 : (i32) -> tensor<1x16xi32>
        %23 = arith.muli %21, %22 : tensor<1x16xi32>
        %24 = tt.broadcast %20 : (tensor<16x1xi32>) -> tensor<16x16xi32>
        %25 = tt.broadcast %23 : (tensor<1x16xi32>) -> tensor<16x16xi32>
        %26 = arith.addi %24, %25 : tensor<16x16xi32>
        %27 = tt.splat %arg0 : (!tt.ptr<f32>) -> tensor<16x16x!tt.ptr<f32>>
        %28 = tt.addptr %27, %26 : tensor<16x16x!tt.ptr<f32>>, tensor<16x16xi32>
        %29 = tt.expand_dims %12 {axis = 1 : i32} : (tensor<16xi32>) -> tensor<16x1xi32>
        %30 = tt.splat %arg8 : (i32) -> tensor<16x1xi32>
        %31 = arith.muli %29, %30 : tensor<16x1xi32>
        %32 = tt.expand_dims %17 {axis = 0 : i32} : (tensor<16xi32>) -> tensor<1x16xi32>
        %33 = tt.splat %arg9 : (i32) -> tensor<1x16xi32>
        %34 = arith.muli %32, %33 : tensor<1x16xi32>
        %35 = tt.broadcast %31 : (tensor<16x1xi32>) -> tensor<16x16xi32>
        %36 = tt.broadcast %34 : (tensor<1x16xi32>) -> tensor<16x16xi32>
        %37 = arith.addi %35, %36 : tensor<16x16xi32>
        %38 = tt.splat %arg1 : (!tt.ptr<f32>) -> tensor<16x16x!tt.ptr<f32>>
        %39 = tt.addptr %38, %37 : tensor<16x16x!tt.ptr<f32>>, tensor<16x16xi32>
        %cst = arith.constant dense<0.000000e+00> : tensor<16x16xf32>
        %40 = arith.divsi %arg5, %c16_i32 : i32
        %c0 = arith.constant 0 : index
        %41 = arith.index_cast %40 : i32 to index
        %c1 = arith.constant 1 : index
        %42:3 = scf.for %arg12 = %c0 to %41 step %c1 iter_args(%arg13 = %cst, %arg14 = %28, %arg15 = %39) -> (tensor<16x16xf32>, tensor<16x16x!tt.ptr<f32>>, tensor<16x16x!tt.ptr<f32>>) {
          %c16 = arith.constant 16 : index
          %59 = arith.muli %arg12, %c16 : index
          %60 = arith.index_cast %59 : index to i32
          %61 = arith.subi %arg5, %60 : i32
          %62 = tt.splat %61 : (i32) -> tensor<1x16xi32>
          %63 = arith.cmpi slt, %21, %62 : tensor<1x16xi32>
          %cst_0 = arith.constant 0.000000e+00 : f32
          %64 = tt.splat %cst_0 : (f32) -> tensor<16x16xf32>
          %65 = tt.broadcast %63 : (tensor<1x16xi1>) -> tensor<16x16xi1>
          %66 = tt.load %28, %65, %64 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<16x16xf32>
          %67 = tt.splat %61 : (i32) -> tensor<16x1xi32>
          %68 = arith.cmpi slt, %29, %67 : tensor<16x1xi32>
          %69 = tt.broadcast %68 : (tensor<16x1xi1>) -> tensor<16x16xi1>
          %70 = tt.load %39, %69, %64 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<16x16xf32>
          %71 = tt.dot %66, %70, %64 {allowTF32 = true} : tensor<16x16xf32> * tensor<16x16xf32> -> tensor<16x16xf32>
          %72 = arith.addf %arg13, %71 : tensor<16x16xf32>
          %73 = arith.muli %arg7, %c16_i32 : i32
          %74 = tt.splat %73 : (i32) -> tensor<16x16xi32>
          %75 = tt.addptr %arg14, %74 : tensor<16x16x!tt.ptr<f32>>, tensor<16x16xi32>
          %76 = arith.muli %arg8, %c16_i32 : i32
          %77 = tt.splat %76 : (i32) -> tensor<16x16xi32>
          %78 = tt.addptr %arg15, %77 : tensor<16x16x!tt.ptr<f32>>, tensor<16x16xi32>
          scf.yield %72, %75, %78 : tensor<16x16xf32>, tensor<16x16x!tt.ptr<f32>>, tensor<16x16x!tt.ptr<f32>>
        }
        %43 = tt.splat %arg10 : (i32) -> tensor<16x1xi32>
        %44 = arith.muli %43, %18 : tensor<16x1xi32>
        %45 = tt.splat %arg2 : (!tt.ptr<f32>) -> tensor<16x1x!tt.ptr<f32>>
        %46 = tt.addptr %45, %44 : tensor<16x1x!tt.ptr<f32>>, tensor<16x1xi32>
        %47 = tt.splat %arg11 : (i32) -> tensor<1x16xi32>
        %48 = arith.muli %47, %32 : tensor<1x16xi32>
        %49 = tt.broadcast %46 : (tensor<16x1x!tt.ptr<f32>>) -> tensor<16x16x!tt.ptr<f32>>
        %50 = tt.broadcast %48 : (tensor<1x16xi32>) -> tensor<16x16xi32>
        %51 = tt.addptr %49, %50 : tensor<16x16x!tt.ptr<f32>>, tensor<16x16xi32>
        %52 = tt.splat %arg3 : (i32) -> tensor<16x1xi32>
        %53 = arith.cmpi slt, %18, %52 : tensor<16x1xi32>
        %54 = tt.splat %arg4 : (i32) -> tensor<1x16xi32>
        %55 = arith.cmpi slt, %32, %54 : tensor<1x16xi32>
        %56 = tt.broadcast %53 : (tensor<16x1xi1>) -> tensor<16x16xi1>
        %57 = tt.broadcast %55 : (tensor<1x16xi1>) -> tensor<16x16xi1>
        %58 = arith.andi %56, %57 : tensor<16x16xi1>
        tt.store %51, %42#0, %58 {cache = 1 : i32, evict = 1 : i32} : tensor<16x16xf32>
        tt.return
      }
    }
    """
    )

    filecheck(correct, ctx.module)
