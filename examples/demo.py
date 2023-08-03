import ctypes
from pathlib import Path

import numpy as np
from mlir_utils import DefaultContext
from mlir_utils.dialects.bufferization import to_memref
from mlir_utils.dialects.ext import arith
from mlir_utils.dialects.ext.scf import yield_, range_
from mlir_utils.dialects.memref import copy
from mlir_utils.runtime.passes import Pipeline, convert_linalg_to_loops
from mlir_utils.runtime.refbackend import LLVMJITBackend

# noinspection PyUnresolvedReferences
from mlir_utils.testing import mlir_ctx as ctx, filecheck, MLIRContext, backend
from mlir_utils.types import i32_t
from mlir_utils.util import find_ops
from triton_mlir_bindings.runtime import get_unranked_memref_descriptor
from triton_mlir_bindings.util.utils import mlir_mod_ctx

from triton_air.dialects.ext import triton as tl
from triton_air.types import p_f32_t, p_f64_t, float64

# needed since the fix isn't defined here nor conftest.py

THIS_DIR = str(Path(__file__).parent.absolute())


def get_asm(operation):
    return operation.get_asm(enable_debug_info=True, pretty_debug_info=True).replace(
        THIS_DIR, "THIS_DIR"
    )


def vadd_run(ctx: MLIRContext, backend: LLVMJITBackend):
    BLOCK_SIZE = 64

    @tl.jit
    def vadd(x_ptr: p_f32_t, y_ptr: p_f32_t, output_ptr: p_f32_t, n_elements: i32_t):
        pid = tl.program_id(axis="x")
        block_size = arith.constant(BLOCK_SIZE, i32_t)
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


def matmul_run(ctx: MLIRContext, backend: LLVMJITBackend):
    D = 32
    BLOCK_SIZE_M = D
    BLOCK_SIZE_K = D
    BLOCK_SIZE_N = D
    GROUP_SIZE_M = 1

    @tl.jit
    def matmul_kernel(
        a_ptr: p_f64_t,
        b_ptr: p_f64_t,
        c_ptr: p_f64_t,
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

        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=float64)
        acc = accumulator

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

    a = np.ones((M, K)).astype(np.float64)
    b = np.ones((K, N)).astype(np.float64)
    c = np.zeros((M, N)).astype(np.float64)

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
    assert np.allclose(r, c)


for i in range(1):
    print(i)
    with mlir_mod_ctx(allow_unregistered_dialects=True, context=DefaultContext) as ctx:
        matmul_run(ctx, LLVMJITBackend())
    # vadd(ctx, LLVMJITBackend())
