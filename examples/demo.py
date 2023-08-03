from mlir_utils import DefaultContext
from triton_mlir_bindings.util.utils import mlir_mod_ctx

from triton_air.dialects import air


import ctypes
from textwrap import dedent

import numpy as np
import pytest
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
from triton_mlir_bindings._mlir_libs._mlir.passmanager import PassManager
from triton_mlir_bindings.runtime import get_unranked_memref_descriptor

from triton_air.dialects.ext import triton as tl
from triton_air.types import p_f32_t, float32

# needed since the fix isn't defined here nor conftest.py


def matmul(ctx: MLIRContext, backend: LLVMJITBackend):
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

    module = backend.compile(
        ctx.module,
        kernel_name="matmul_kernel",
        pipeline=Pipeline().add_pass("triton-to-linalg"),
        generate_kernel_wrapper=False,
        generate_return_consumer=False,
    )

    print(module)


with mlir_mod_ctx(allow_unregistered_dialects=True, context=DefaultContext) as ctx:
    matmul(ctx, LLVMJITBackend())
