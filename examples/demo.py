import ctypes
from textwrap import dedent

import numpy as np
from mlir_utils.dialects.bufferization import to_memref
from mlir_utils.dialects.ext import arith
from mlir_utils.dialects.ext.scf import yield_, range_
from mlir_utils.dialects.memref import copy
from mlir_utils.runtime.passes import Pipeline, convert_linalg_to_loops
from mlir_utils.runtime.refbackend import LLVMJITBackend

# noinspection PyUnresolvedReferences
from mlir_utils.testing import filecheck, MLIRContext, backend
from mlir_utils.util import find_ops
from triton_mlir_bindings.runtime import get_unranked_memref_descriptor

import triton_pp.types as T
from triton_pp.dialects.ext import triton as tl
from triton_pp.util import mlir_ctx_man


def vadd_lower_to_linalg(ctx: MLIRContext, backend: LLVMJITBackend):
    BLOCK_SIZE = 64

    @tl.jit
    def vadd(
        x_ptr: T.p_f32_t, y_ptr: T.p_f32_t, output_ptr: T.p_f32_t, n_elements: T.int32
    ):
        pid = tl.program_id(axis="x")
        block_size = arith.constant(BLOCK_SIZE, T.int32)
        block_start = pid * block_size
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements

        x = tl.load(x_ptr + offsets, mask)
        y = tl.load(y_ptr + offsets, mask)

        output = x + y
        tl.store(output_ptr + offsets, output, mask)

    vadd.emit()

    module = backend.compile(
        ctx.module,
        kernel_name="vadd",
        pipeline=Pipeline().add_pass("triton-to-linalg").cse(),
        generate_kernel_wrapper=False,
        generate_return_consumer=False,
    )

    correct = dedent(
        """\
    #map = affine_map<(d0) -> (d0)>
    module {
      func.func @vadd(%arg0: memref<*xf32>, %arg1: memref<*xf32>, %arg2: memref<*xf32>, %arg3: i32, %arg4: i32, %arg5: i32, %arg6: i32) {
        %c64 = arith.constant 64 : index
        %c64_i32 = arith.constant 64 : i32
        %0 = arith.muli %arg4, %c64_i32 : i32
        %1 = arith.index_cast %0 : i32 to index
        %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [%1], sizes: [64], strides: [1] : memref<*xf32> to memref<64xf32, strided<[1], offset: ?>>
        %alloc = memref.alloc() : memref<64xf32>
        %2 = arith.addi %1, %c64 : index
        %3 = arith.index_cast %arg3 : i32 to index
        %4 = arith.minsi %2, %3 : index
        %5 = arith.subi %4, %1 : index
        %subview = memref.subview %reinterpret_cast[0] [%5] [1] : memref<64xf32, strided<[1], offset: ?>> to memref<?xf32, strided<[1], offset: ?>>
        %subview_0 = memref.subview %alloc[0] [%5] [1] : memref<64xf32> to memref<?xf32, strided<[1]>>
        memref.copy %subview, %subview_0 : memref<?xf32, strided<[1], offset: ?>> to memref<?xf32, strided<[1]>>
        %6 = bufferization.to_tensor %alloc restrict writable : memref<64xf32>
        %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%1], sizes: [64], strides: [1] : memref<*xf32> to memref<64xf32, strided<[1], offset: ?>>
        %alloc_2 = memref.alloc() : memref<64xf32>
        %subview_3 = memref.subview %reinterpret_cast_1[0] [%5] [1] : memref<64xf32, strided<[1], offset: ?>> to memref<?xf32, strided<[1], offset: ?>>
        %subview_4 = memref.subview %alloc_2[0] [%5] [1] : memref<64xf32> to memref<?xf32, strided<[1]>>
        memref.copy %subview_3, %subview_4 : memref<?xf32, strided<[1], offset: ?>> to memref<?xf32, strided<[1]>>
        %7 = bufferization.to_tensor %alloc_2 restrict writable : memref<64xf32>
        %8 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins(%6, %7 : tensor<64xf32>, tensor<64xf32>) outs(%6 : tensor<64xf32>) {
        ^bb0(%in: f32, %in_7: f32, %out: f32):
          %9 = arith.addf %in, %in_7 : f32
          linalg.yield %9 : f32
        } -> tensor<64xf32>
        %reinterpret_cast_5 = memref.reinterpret_cast %arg2 to offset: [%1], sizes: [64], strides: [1] : memref<*xf32> to memref<64xf32, strided<[1], offset: ?>>
        %extracted_slice = tensor.extract_slice %8[0] [%5] [1] : tensor<64xf32> to tensor<?xf32>
        %subview_6 = memref.subview %reinterpret_cast_5[0] [%5] [1] : memref<64xf32, strided<[1], offset: ?>> to memref<?xf32, strided<[1], offset: ?>>
        memref.tensor_store %extracted_slice, %subview_6 : memref<?xf32, strided<[1], offset: ?>>
        return
      }
    }
    """
    )
    filecheck(correct, module)


def vadd_run(ctx: MLIRContext, backend: LLVMJITBackend):
    BLOCK_SIZE = 64

    @tl.jit
    def vadd(
        x_ptr: T.p_f32_t, y_ptr: T.p_f32_t, output_ptr: T.p_f32_t, n_elements: T.int32
    ):
        pid = tl.program_id(axis="x")
        block_size = arith.constant(BLOCK_SIZE, T.int32)
        block_start = pid * block_size
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements

        x = tl.load(x_ptr + offsets, mask)
        y = tl.load(y_ptr + offsets, mask)

        output = x + y
        tl.store(output_ptr + offsets, output, mask)

    vadd.emit()

    module = backend.compile(
        ctx.module,
        kernel_name="vadd",
        pipeline=Pipeline().add_pass("triton-to-linalg"),
        generate_kernel_wrapper=False,
        generate_return_consumer=False,
    )

    tensor_store = find_ops(
        module.operation,
        lambda op: op.operation.name == "memref.tensor_store",
        single=True,
    )
    memref = to_memref(tensor_store.memref.type, tensor_store.tensor)
    memref.owner.move_after(tensor_store)
    c = copy(memref, tensor_store.memref)
    c.move_after(memref.owner)
    tensor_store.operation.erase()

    module = backend.compile(
        module,
        kernel_name="vadd",
        pipeline=Pipeline().bufferize().Func(convert_linalg_to_loops()).lower_to_llvm(),
        generate_kernel_wrapper=False,
        generate_return_consumer=False,
    )
    correct = dedent(
        """\
    module attributes {llvm.data_layout = ""} {
      llvm.func @free(!llvm.ptr)
      llvm.func @memrefCopy(i64, !llvm.ptr, !llvm.ptr)
      llvm.func @malloc(i64) -> !llvm.ptr
      llvm.func @vadd(%arg0: i64, %arg1: !llvm.ptr, %arg2: i64, %arg3: !llvm.ptr, %arg4: i64, %arg5: !llvm.ptr, %arg6: i32, %arg7: i32, %arg8: i32, %arg9: i32) attributes {llvm.emit_c_interface} {
        %0 = llvm.mlir.constant(4 : index) : i64
        %1 = llvm.mlir.constant(1 : i64) : i64
        %2 = llvm.mlir.constant(64 : i32) : i32
        %3 = llvm.mlir.constant(1 : index) : i64
        %4 = llvm.mlir.constant(64 : index) : i64
        %5 = llvm.mlir.constant(0 : index) : i64
        %6 = llvm.mul %arg7, %2  : i32
        %7 = llvm.sext %6 : i32 to i64
        %8 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
        %9 = llvm.load %arg1 : !llvm.ptr -> !llvm.ptr
        %10 = llvm.getelementptr %arg1[1] : (!llvm.ptr) -> !llvm.ptr, !llvm.ptr
        %11 = llvm.load %10 : !llvm.ptr -> !llvm.ptr
        %12 = llvm.mlir.null : !llvm.ptr
        %13 = llvm.getelementptr %12[64] : (!llvm.ptr) -> !llvm.ptr, f32
        %14 = llvm.ptrtoint %13 : !llvm.ptr to i64
        %15 = llvm.call @malloc(%14) : (i64) -> !llvm.ptr
        %16 = llvm.add %7, %4  : i64
        %17 = llvm.sext %arg6 : i32 to i64
        %18 = llvm.intr.smin(%16, %17)  : (i64, i64) -> i64
        %19 = llvm.sub %18, %7  : i64
        %20 = llvm.insertvalue %9, %8[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
        %21 = llvm.insertvalue %11, %20[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
        %22 = llvm.insertvalue %7, %21[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
        %23 = llvm.insertvalue %19, %22[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
        %24 = llvm.insertvalue %3, %23[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
        %25 = llvm.insertvalue %15, %8[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
        %26 = llvm.insertvalue %15, %25[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
        %27 = llvm.insertvalue %5, %26[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
        %28 = llvm.insertvalue %19, %27[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
        %29 = llvm.insertvalue %3, %28[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
        %30 = llvm.intr.stacksave : !llvm.ptr
        %31 = llvm.alloca %3 x !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> : (i64) -> !llvm.ptr
        llvm.store %24, %31 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.ptr
        %32 = llvm.mlir.undef : !llvm.struct<(i64, ptr)>
        %33 = llvm.insertvalue %1, %32[0] : !llvm.struct<(i64, ptr)>
        %34 = llvm.insertvalue %31, %33[1] : !llvm.struct<(i64, ptr)>
        %35 = llvm.alloca %3 x !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> : (i64) -> !llvm.ptr
        llvm.store %29, %35 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.ptr
        %36 = llvm.insertvalue %35, %33[1] : !llvm.struct<(i64, ptr)>
        %37 = llvm.alloca %3 x !llvm.struct<(i64, ptr)> : (i64) -> !llvm.ptr
        llvm.store %34, %37 : !llvm.struct<(i64, ptr)>, !llvm.ptr
        %38 = llvm.alloca %3 x !llvm.struct<(i64, ptr)> : (i64) -> !llvm.ptr
        llvm.store %36, %38 : !llvm.struct<(i64, ptr)>, !llvm.ptr
        llvm.call @memrefCopy(%0, %37, %38) : (i64, !llvm.ptr, !llvm.ptr) -> ()
        llvm.intr.stackrestore %30 : !llvm.ptr
        %39 = llvm.load %arg3 : !llvm.ptr -> !llvm.ptr
        %40 = llvm.getelementptr %arg3[1] : (!llvm.ptr) -> !llvm.ptr, !llvm.ptr
        %41 = llvm.load %40 : !llvm.ptr -> !llvm.ptr
        %42 = llvm.call @malloc(%14) : (i64) -> !llvm.ptr
        %43 = llvm.insertvalue %39, %8[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
        %44 = llvm.insertvalue %41, %43[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
        %45 = llvm.insertvalue %7, %44[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
        %46 = llvm.insertvalue %19, %45[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
        %47 = llvm.insertvalue %3, %46[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
        %48 = llvm.insertvalue %42, %8[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
        %49 = llvm.insertvalue %42, %48[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
        %50 = llvm.insertvalue %5, %49[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
        %51 = llvm.insertvalue %19, %50[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
        %52 = llvm.insertvalue %3, %51[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
        %53 = llvm.intr.stacksave : !llvm.ptr
        %54 = llvm.alloca %3 x !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> : (i64) -> !llvm.ptr
        llvm.store %47, %54 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.ptr
        %55 = llvm.insertvalue %54, %33[1] : !llvm.struct<(i64, ptr)>
        %56 = llvm.alloca %3 x !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> : (i64) -> !llvm.ptr
        llvm.store %52, %56 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.ptr
        %57 = llvm.insertvalue %56, %33[1] : !llvm.struct<(i64, ptr)>
        %58 = llvm.alloca %3 x !llvm.struct<(i64, ptr)> : (i64) -> !llvm.ptr
        llvm.store %55, %58 : !llvm.struct<(i64, ptr)>, !llvm.ptr
        %59 = llvm.alloca %3 x !llvm.struct<(i64, ptr)> : (i64) -> !llvm.ptr
        llvm.store %57, %59 : !llvm.struct<(i64, ptr)>, !llvm.ptr
        llvm.call @memrefCopy(%0, %58, %59) : (i64, !llvm.ptr, !llvm.ptr) -> ()
        llvm.intr.stackrestore %53 : !llvm.ptr
        %60 = llvm.add %14, %4  : i64
        %61 = llvm.call @malloc(%60) : (i64) -> !llvm.ptr
        %62 = llvm.ptrtoint %61 : !llvm.ptr to i64
        %63 = llvm.sub %4, %3  : i64
        %64 = llvm.add %62, %63  : i64
        %65 = llvm.urem %64, %4  : i64
        %66 = llvm.sub %64, %65  : i64
        %67 = llvm.inttoptr %66 : i64 to !llvm.ptr
        llvm.br ^bb1(%5 : i64)
      ^bb1(%68: i64):  // 2 preds: ^bb0, ^bb2
        %69 = llvm.icmp "slt" %68, %4 : i64
        llvm.cond_br %69, ^bb2, ^bb3
      ^bb2:  // pred: ^bb1
        %70 = llvm.getelementptr %15[%68] : (!llvm.ptr, i64) -> !llvm.ptr, f32
        %71 = llvm.load %70 : !llvm.ptr -> f32
        %72 = llvm.getelementptr %42[%68] : (!llvm.ptr, i64) -> !llvm.ptr, f32
        %73 = llvm.load %72 : !llvm.ptr -> f32
        %74 = llvm.fadd %71, %73  : f32
        %75 = llvm.getelementptr %67[%68] : (!llvm.ptr, i64) -> !llvm.ptr, f32
        llvm.store %74, %75 : f32, !llvm.ptr
        %76 = llvm.add %68, %3  : i64
        llvm.br ^bb1(%76 : i64)
      ^bb3:  // pred: ^bb1
        llvm.call @free(%42) : (!llvm.ptr) -> ()
        llvm.call @free(%15) : (!llvm.ptr) -> ()
        %77 = llvm.load %arg5 : !llvm.ptr -> !llvm.ptr
        %78 = llvm.getelementptr %arg5[1] : (!llvm.ptr) -> !llvm.ptr, !llvm.ptr
        %79 = llvm.load %78 : !llvm.ptr -> !llvm.ptr
        %80 = llvm.insertvalue %61, %8[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
        %81 = llvm.insertvalue %67, %80[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
        %82 = llvm.insertvalue %5, %81[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
        %83 = llvm.insertvalue %19, %82[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
        %84 = llvm.insertvalue %3, %83[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
        %85 = llvm.insertvalue %77, %8[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
        %86 = llvm.insertvalue %79, %85[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
        %87 = llvm.insertvalue %7, %86[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
        %88 = llvm.insertvalue %19, %87[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
        %89 = llvm.insertvalue %3, %88[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
        %90 = llvm.intr.stacksave : !llvm.ptr
        %91 = llvm.alloca %3 x !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> : (i64) -> !llvm.ptr
        llvm.store %84, %91 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.ptr
        %92 = llvm.insertvalue %91, %33[1] : !llvm.struct<(i64, ptr)>
        %93 = llvm.alloca %3 x !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> : (i64) -> !llvm.ptr
        llvm.store %89, %93 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.ptr
        %94 = llvm.insertvalue %93, %33[1] : !llvm.struct<(i64, ptr)>
        %95 = llvm.alloca %3 x !llvm.struct<(i64, ptr)> : (i64) -> !llvm.ptr
        llvm.store %92, %95 : !llvm.struct<(i64, ptr)>, !llvm.ptr
        %96 = llvm.alloca %3 x !llvm.struct<(i64, ptr)> : (i64) -> !llvm.ptr
        llvm.store %94, %96 : !llvm.struct<(i64, ptr)>, !llvm.ptr
        llvm.call @memrefCopy(%0, %95, %96) : (i64, !llvm.ptr, !llvm.ptr) -> ()
        llvm.intr.stackrestore %90 : !llvm.ptr
        llvm.call @free(%61) : (!llvm.ptr) -> ()
        llvm.return
      }
      llvm.func @_mlir_ciface_vadd(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: !llvm.ptr, %arg3: i32, %arg4: i32, %arg5: i32, %arg6: i32) attributes {llvm.emit_c_interface} {
        %0 = llvm.load %arg0 : !llvm.ptr -> !llvm.struct<(i64, ptr)>
        %1 = llvm.extractvalue %0[0] : !llvm.struct<(i64, ptr)>
        %2 = llvm.extractvalue %0[1] : !llvm.struct<(i64, ptr)>
        %3 = llvm.load %arg1 : !llvm.ptr -> !llvm.struct<(i64, ptr)>
        %4 = llvm.extractvalue %3[0] : !llvm.struct<(i64, ptr)>
        %5 = llvm.extractvalue %3[1] : !llvm.struct<(i64, ptr)>
        %6 = llvm.load %arg2 : !llvm.ptr -> !llvm.struct<(i64, ptr)>
        %7 = llvm.extractvalue %6[0] : !llvm.struct<(i64, ptr)>
        %8 = llvm.extractvalue %6[1] : !llvm.struct<(i64, ptr)>
        llvm.call @vadd(%1, %2, %4, %5, %7, %8, %arg3, %arg4, %arg5, %arg6) : (i64, !llvm.ptr, i64, !llvm.ptr, i64, !llvm.ptr, i32, i32, i32, i32) -> ()
        llvm.return
      }
    }
    """
    )

    filecheck(correct, module)

    n_elements = 64
    a = np.ones((n_elements,)).astype(np.float32)
    b = np.ones((n_elements,)).astype(np.float32)
    c = np.zeros((n_elements,)).astype(np.float32)
    A = ctypes.pointer(ctypes.pointer(get_unranked_memref_descriptor(a)))
    B = ctypes.pointer(ctypes.pointer(get_unranked_memref_descriptor(b)))
    C = ctypes.pointer(ctypes.pointer(get_unranked_memref_descriptor(c)))
    n_elements_ = ctypes.c_int(n_elements)
    # all zero
    launch_grid_x = ctypes.c_int()
    launch_grid_y = ctypes.c_int()
    launch_grid_z = ctypes.c_int()

    invoker = backend.load(module)
    invoker.ee.invoke(
        "vadd",
        A,
        B,
        C,
        ctypes.byref(n_elements_),
        ctypes.byref(launch_grid_x),
        ctypes.byref(launch_grid_y),
        ctypes.byref(launch_grid_z),
    )
    assert np.array_equal(a + b, c)


def _matmul(ctx: MLIRContext, backend: LLVMJITBackend):
    BLOCK_SIZE_M = 16
    BLOCK_SIZE_N = 16
    BLOCK_SIZE_K = 16
    GROUP_SIZE_M = 2

    @tl.jit
    def matmul_kernel(
        a_ptr: T.p_f32_t,
        b_ptr: T.p_f32_t,
        c_ptr: T.p_f32_t,
        M: T.int32,
        N: T.int32,
        K: T.int32,
        stride_am: T.int32,
        stride_ak: T.int32,
        stride_bk: T.int32,
        stride_bn: T.int32,
        stride_cm: T.int32,
        stride_cn: T.int32,
    ):
        pid = tl.program_id(axis="x")
        num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
        num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
        num_pid_in_group = GROUP_SIZE_M * num_pid_n
        group_id = pid // num_pid_in_group
        first_pid_m = group_id * GROUP_SIZE_M
        # TODO(max): min isn't doing anything here
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

        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=T.float32)
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
    module = backend.compile(
        ctx.module,
        kernel_name="matmul_kernel",
        pipeline=Pipeline().add_pass("triton-to-linalg"),
        generate_kernel_wrapper=False,
        generate_return_consumer=False,
    )

    correct = dedent(
        """\
    #map = affine_map<(d0, d1) -> (d0, d1)>
    module {
      func.func @matmul_kernel(%arg0: memref<*xf32>, %arg1: memref<*xf32>, %arg2: memref<*xf32>, %arg3: i32, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32, %arg9: i32, %arg10: i32, %arg11: i32, %arg12: i32, %arg13: i32, %arg14: i32) {
        %c16_i32 = arith.constant 16 : i32
        %c2_i32 = arith.constant 2 : i32
        %c16 = arith.constant 16 : index
        %c1 = arith.constant 1 : index
        %c0 = arith.constant 0 : index
        %cst = arith.constant 0.000000e+00 : f32
        %0 = tensor.empty() : tensor<16x16xf32>
        %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<16x16xf32>) -> tensor<16x16xf32>
        %2 = arith.divsi %arg4, %c16_i32 : i32
        %3 = arith.muli %2, %c2_i32 : i32
        %4 = arith.floordivsi %arg12, %3 : i32
        %5 = arith.muli %4, %c2_i32 : i32
        %6 = arith.remsi %arg12, %c2_i32 : i32
        %7 = arith.addi %5, %6 : i32
        %8 = arith.remsi %arg12, %3 : i32
        %9 = arith.floordivsi %8, %c2_i32 : i32
        %10 = arith.muli %7, %c16_i32 : i32
        %11 = arith.muli %9, %c16_i32 : i32
        %12 = arith.index_cast %10 : i32 to index
        %13 = arith.index_cast %arg6 : i32 to index
        %14 = arith.muli %12, %13 : index
        %15 = arith.index_cast %arg7 : i32 to index
        %16 = arith.index_cast %arg8 : i32 to index
        %17 = arith.index_cast %11 : i32 to index
        %18 = arith.index_cast %arg9 : i32 to index
        %19 = arith.muli %17, %18 : index
        %20 = arith.divsi %arg5, %c16_i32 : i32
        %21 = arith.index_cast %20 : i32 to index
        %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [%14], sizes: [16, 16], strides: [%13, %15] : memref<*xf32> to memref<16x16xf32, strided<[?, ?], offset: ?>>
        %reinterpret_cast_0 = memref.reinterpret_cast %arg1 to offset: [%19], sizes: [16, 16], strides: [%16, %18] : memref<*xf32> to memref<16x16xf32, strided<[?, ?], offset: ?>>
        %22:5 = scf.for %arg15 = %c0 to %21 step %c1 iter_args(%arg16 = %1, %arg17 = %14, %arg18 = %c0, %arg19 = %19, %arg20 = %c0) -> (tensor<16x16xf32>, index, index, index, index) {
          %44 = arith.muli %arg15, %c16 : index
          %45 = arith.index_cast %44 : index to i32
          %46 = arith.subi %arg5, %45 : i32
          %alloc = memref.alloc() : memref<16x16xf32>
          %47 = arith.index_cast %46 : i32 to index
          %48 = arith.minsi %47, %c16 : index
          %subview_2 = memref.subview %reinterpret_cast[0, 0] [16, %48] [1, 1] : memref<16x16xf32, strided<[?, ?], offset: ?>> to memref<16x?xf32, strided<[?, ?], offset: ?>>
          %subview_3 = memref.subview %alloc[0, 0] [16, %48] [1, 1] : memref<16x16xf32> to memref<16x?xf32, strided<[16, 1]>>
          %49 = arith.cmpi slt, %48, %c16 : index
          scf.if %49 {
            linalg.fill ins(%cst : f32) outs(%alloc : memref<16x16xf32>)
          }
          memref.copy %subview_2, %subview_3 : memref<16x?xf32, strided<[?, ?], offset: ?>> to memref<16x?xf32, strided<[16, 1]>>
          %50 = bufferization.to_tensor %alloc restrict writable : memref<16x16xf32>
          %51 = arith.muli %arg15, %c16 : index
          %52 = arith.index_cast %51 : index to i32
          %53 = arith.subi %arg5, %52 : i32
          %alloc_4 = memref.alloc() : memref<16x16xf32>
          %54 = arith.index_cast %53 : i32 to index
          %55 = arith.minsi %54, %c16 : index
          %subview_5 = memref.subview %reinterpret_cast_0[0, 0] [%55, 16] [1, 1] : memref<16x16xf32, strided<[?, ?], offset: ?>> to memref<?x16xf32, strided<[?, ?], offset: ?>>
          %subview_6 = memref.subview %alloc_4[0, 0] [%55, 16] [1, 1] : memref<16x16xf32> to memref<?x16xf32, strided<[16, 1]>>
          %56 = arith.cmpi slt, %55, %c16 : index
          scf.if %56 {
            linalg.fill ins(%cst : f32) outs(%alloc_4 : memref<16x16xf32>)
          }
          memref.copy %subview_5, %subview_6 : memref<?x16xf32, strided<[?, ?], offset: ?>> to memref<?x16xf32, strided<[16, 1]>>
          %57 = bufferization.to_tensor %alloc_4 restrict writable : memref<16x16xf32>
          %58 = tensor.empty() : tensor<16x16xf32>
          %59 = linalg.matmul ins(%50, %57 : tensor<16x16xf32>, tensor<16x16xf32>) outs(%58 : tensor<16x16xf32>) -> tensor<16x16xf32>
          %60 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%59, %1 : tensor<16x16xf32>, tensor<16x16xf32>) outs(%59 : tensor<16x16xf32>) {
          ^bb0(%in: f32, %in_7: f32, %out: f32):
            %70 = arith.addf %in, %in_7 : f32
            linalg.yield %70 : f32
          } -> tensor<16x16xf32>
          %61 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%arg16, %60 : tensor<16x16xf32>, tensor<16x16xf32>) outs(%arg16 : tensor<16x16xf32>) {
          ^bb0(%in: f32, %in_7: f32, %out: f32):
            %70 = arith.addf %in, %in_7 : f32
            linalg.yield %70 : f32
          } -> tensor<16x16xf32>
          %62 = arith.muli %arg7, %c16_i32 : i32
          %63 = arith.index_cast %62 : i32 to index
          %64 = arith.addi %arg17, %63 : index
          %65 = arith.addi %64, %arg18 : index
          %66 = arith.muli %arg8, %c16_i32 : i32
          %67 = arith.index_cast %66 : i32 to index
          %68 = arith.addi %arg19, %67 : index
          %69 = arith.addi %68, %arg20 : index
          scf.yield %61, %65, %c0, %69, %c0 : tensor<16x16xf32>, index, index, index, index
        }
        %23 = arith.muli %7, %c16_i32 : i32
        %24 = arith.muli %9, %c16_i32 : i32
        %25 = arith.index_cast %arg10 : i32 to index
        %26 = arith.index_cast %23 : i32 to index
        %27 = arith.muli %26, %25 : index
        %28 = arith.index_cast %arg11 : i32 to index
        %29 = arith.index_cast %24 : i32 to index
        %30 = arith.muli %29, %28 : index
        %31 = arith.addi %27, %30 : index
        %reinterpret_cast_1 = memref.reinterpret_cast %arg2 to offset: [%31], sizes: [16, 16], strides: [%25, %28] : memref<*xf32> to memref<16x16xf32, strided<[?, ?], offset: ?>>
        %32 = arith.index_cast %23 : i32 to index
        %33 = arith.addi %32, %c16 : index
        %34 = arith.index_cast %arg3 : i32 to index
        %35 = arith.minsi %33, %34 : index
        %36 = arith.subi %35, %32 : index
        %37 = arith.index_cast %24 : i32 to index
        %38 = arith.addi %37, %c16 : index
        %39 = arith.index_cast %arg4 : i32 to index
        %40 = arith.minsi %38, %39 : index
        %41 = arith.subi %40, %37 : index
        %42 = arith.minsi %36, %c16 : index
        %43 = arith.minsi %41, %c16 : index
        %extracted_slice = tensor.extract_slice %22#0[0, 0] [%42, %43] [1, 1] : tensor<16x16xf32> to tensor<?x?xf32>
        %subview = memref.subview %reinterpret_cast_1[0, 0] [%42, %43] [1, 1] : memref<16x16xf32, strided<[?, ?], offset: ?>> to memref<?x?xf32, strided<[?, ?], offset: ?>>
        memref.tensor_store %extracted_slice, %subview : memref<?x?xf32, strided<[?, ?], offset: ?>>
        return
      }
    }
    """
    )

    filecheck(correct, module)


def matmul_run(ctx: MLIRContext, backend: LLVMJITBackend):
    D = 4
    BLOCK_SIZE_M = D
    BLOCK_SIZE_K = D
    BLOCK_SIZE_N = D
    GROUP_SIZE_M = 1

    @tl.jit
    def matmul_kernel(
        a_ptr: +T.int32,
        b_ptr: +T.int32,
        c_ptr: +T.int32,
        M: T.int32,
        N: T.int32,
        K: T.int32,
        stride_am: T.int32,
        stride_ak: T.int32,
        stride_bk: T.int32,
        stride_bn: T.int32,
        stride_cm: T.int32,
        stride_cn: T.int32,
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

        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=T.int32)
        acc = accumulator

        # r = tl.cdiv(K, BLOCK_SIZE_K)
        r = 1
        for k, (acc, aptrs, bptrs) in range_(
            0, r, iter_args=[accumulator, a_ptrs, b_ptrs]
        ):
            mask = offs_k[None, :] < K - k * BLOCK_SIZE_K
            a = tl.load(a_ptrs, mask=mask, other=0)
            mask = offs_k[:, None] < K - k * BLOCK_SIZE_K
            b = tl.load(b_ptrs, mask=mask, other=0)
            acc += tl.dot(a, b)
            aptrs += BLOCK_SIZE_K * stride_ak
            bptrs += BLOCK_SIZE_K * stride_bk
            acc, *_ = yield_(acc, aptrs, bptrs)

        c = acc

        offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
        c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
        tl.store(c_ptrs, c, mask=c_mask)

    matmul_kernel.emit()
    module = backend.compile(
        ctx.module,
        kernel_name="matmul_kernel",
        pipeline=Pipeline().add_pass("triton-to-linalg"),
        generate_kernel_wrapper=False,
        generate_return_consumer=False,
        verify=False,
    )

    addf = find_ops(
        module.operation,
        lambda op: op.operation.name == "arith.addf",
        single=True,
    )
    addi = arith.addi(addf.operands[0], addf.operands[1], loc=addf.location)
    addi.owner.move_after(addf)
    addf.result.replace_all_uses_with(addi)
    addf.operation.erase()

    tensor_store = find_ops(
        module.operation,
        lambda op: op.operation.name == "memref.tensor_store",
        single=True,
    )
    memref = to_memref(tensor_store.memref.type, tensor_store.tensor)
    memref.owner.move_after(tensor_store)
    c = copy(memref, tensor_store.memref)
    c.move_after(memref.owner)
    tensor_store.operation.erase()

    module = backend.compile(
        module,
        kernel_name="matmul_kernel",
        pipeline=Pipeline()
        .bufferize()
        .Func(
            convert_linalg_to_loops()
            .buffer_loop_hoisting()
            .convert_bufferization_to_memref()
        )
        .lower_to_llvm(),
        generate_kernel_wrapper=False,
        generate_return_consumer=False,
    )

    M = D
    K = D
    N = D

    stride_am = M
    stride_ak = 1
    stride_bk = K
    stride_bn = 1
    stride_cm = M
    stride_cn = 1

    a = np.ones((M, K)).astype(np.int32)
    b = np.ones((K, N)).astype(np.int32)
    c = np.zeros((M, N)).astype(np.int32)

    A = ctypes.pointer(ctypes.pointer(get_unranked_memref_descriptor(a)))
    B = ctypes.pointer(ctypes.pointer(get_unranked_memref_descriptor(b)))
    C = ctypes.pointer(ctypes.pointer(get_unranked_memref_descriptor(c)))

    M_ = ctypes.c_int(M)
    K_ = ctypes.c_int(K)
    N_ = ctypes.c_int(N)

    stride_am_ = ctypes.c_int(stride_am)
    stride_ak_ = ctypes.c_int(stride_ak)
    stride_bk_ = ctypes.c_int(stride_bk)
    stride_bn_ = ctypes.c_int(stride_bn)
    stride_cm_ = ctypes.c_int(stride_cm)
    stride_cn_ = ctypes.c_int(stride_cn)

    launch_grid_x = ctypes.c_int()
    launch_grid_y = ctypes.c_int()
    launch_grid_z = ctypes.c_int()

    invoker = backend.load(module)
    assert len(c.nonzero())
    invoker.ee.invoke(
        "matmul_kernel",
        A,
        B,
        C,
        ctypes.byref(M_),
        ctypes.byref(K_),
        ctypes.byref(N_),
        ctypes.byref(stride_am_),
        ctypes.byref(stride_ak_),
        ctypes.byref(stride_bk_),
        ctypes.byref(stride_bn_),
        ctypes.byref(stride_cm_),
        ctypes.byref(stride_cn_),
        ctypes.byref(launch_grid_x),
        ctypes.byref(launch_grid_y),
        ctypes.byref(launch_grid_z),
    )
    r = a @ b
    assert len(r.nonzero()) > 0
    assert len(c.nonzero()) > 0
    print(r)
    print(c)


for i in range(10):
    print(i)
    with mlir_ctx_man() as ctx:
        matmul_run(ctx, LLVMJITBackend())
