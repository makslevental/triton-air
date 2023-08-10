from mlir_utils.util import get_result_or_results, maybe_cast, region_op
from triton_mlir_bindings.dialects.triton import (
    AddPtrOp,
    AdvanceOp,
    AssertOp,
    AtomicCASOp,
    AtomicRMWOp,
    BitcastOp,
    BroadcastOp,
    CallOp,
    CatOp,
    DotOp,
    ExpandDimsOp,
    ExtElemwiseOp,
    FpToFpOp,
    FuncOp,
    GetNumProgramsOp,
    GetProgramIdOp,
    IntToPtrOp,
    LoadOp,
    MakeRangeOp,
    MakeTensorPtrOp,
    PrintOp,
    PtrToIntOp,
    ReduceOp,
    ReduceReturnOp,
    ReturnOp,
    SplatOp,
    StoreOp,
    TransOp,
    ViewOp,
)
from triton_mlir_bindings.ir import Value, Type


def addptr(result: Type, ptr: Value, offset: Value, *, loc=None, ip=None):
    return maybe_cast(
        get_result_or_results(AddPtrOp(result, ptr, offset, loc=loc, ip=ip))
    )


def advance(result: Type, ptr: Value, offsets: list[Value], *, loc=None, ip=None):
    return maybe_cast(
        get_result_or_results(AdvanceOp(result, ptr, offsets, loc=loc, ip=ip))
    )


def assert_(condition: Value, message, file, func, line, *, loc=None, ip=None):
    return maybe_cast(
        get_result_or_results(
            AssertOp(condition, message, file, func, line, loc=loc, ip=ip)
        )
    )


def atomic_cas(result: Type, ptr: Value, cmp: Value, val: Value, *, loc=None, ip=None):
    return maybe_cast(
        get_result_or_results(AtomicCASOp(result, ptr, cmp, val, loc=loc, ip=ip))
    )


def atomic_rmw(
    result: Type, atomic_rmw_op, ptr: Value, val: Value, *, mask=None, loc=None, ip=None
):
    return maybe_cast(
        get_result_or_results(
            AtomicRMWOp(result, atomic_rmw_op, ptr, val, mask=mask, loc=loc, ip=ip)
        )
    )


def bitcast(result: Type, from_: Value, *, loc=None, ip=None):
    return maybe_cast(get_result_or_results(BitcastOp(result, from_, loc=loc, ip=ip)))


def broadcast(result: Type, src: Value, *, loc=None, ip=None):
    return maybe_cast(get_result_or_results(BroadcastOp(result, src, loc=loc, ip=ip)))


def call(result: list[Type], callee, operands_: list[Value], *, loc=None, ip=None):
    return maybe_cast(
        get_result_or_results(CallOp(result, callee, operands_, loc=loc, ip=ip))
    )


def cat(result: Type, lhs: Value, rhs: Value, *, loc=None, ip=None):
    return maybe_cast(get_result_or_results(CatOp(result, lhs, rhs, loc=loc, ip=ip)))


def dot(a: Value, b: Value, c: Value, allow_tf32, *, loc=None, ip=None):
    return maybe_cast(get_result_or_results(DotOp(a, b, c, allow_tf32, loc=loc, ip=ip)))


def expand_dims(src: Value, axis, *, loc=None, ip=None):
    return maybe_cast(get_result_or_results(ExpandDimsOp(src, axis, loc=loc, ip=ip)))


def ext_elemwise(
    result: Type, args: list[Value], libname, libpath, symbol, *, loc=None, ip=None
):
    return maybe_cast(
        get_result_or_results(
            ExtElemwiseOp(result, args, libname, libpath, symbol, loc=loc, ip=ip)
        )
    )


def fp_to_fp(result: Type, from_: Value, *, loc=None, ip=None):
    return maybe_cast(get_result_or_results(FpToFpOp(result, from_, loc=loc, ip=ip)))


@region_op
def func(
    sym_name,
    function_type,
    *,
    sym_visibility=None,
    arg_attrs=None,
    res_attrs=None,
    loc=None,
    ip=None,
):
    return FuncOp(
        sym_name,
        function_type,
        sym_visibility=sym_visibility,
        arg_attrs=arg_attrs,
        res_attrs=res_attrs,
        loc=loc,
        ip=ip,
    )


def get_num_programs(axis, *, loc=None, ip=None):
    return maybe_cast(get_result_or_results(GetNumProgramsOp(axis, loc=loc, ip=ip)))


def get_program_id(axis, *, loc=None, ip=None):
    return maybe_cast(get_result_or_results(GetProgramIdOp(axis, loc=loc, ip=ip)))


def int_to_ptr(result: Type, from_: Value, *, loc=None, ip=None):
    return maybe_cast(get_result_or_results(IntToPtrOp(result, from_, loc=loc, ip=ip)))


def load(
    result: Type,
    ptr: Value,
    cache,
    evict,
    is_volatile,
    *,
    mask=None,
    other=None,
    boundary_check=None,
    padding=None,
    loc=None,
    ip=None,
):
    return maybe_cast(
        get_result_or_results(
            LoadOp(
                result,
                ptr,
                cache,
                evict,
                is_volatile,
                mask=mask,
                other=other,
                boundaryCheck=boundary_check,
                padding=padding,
                loc=loc,
                ip=ip,
            )
        )
    )


def make_range(result: Type, start, end, *, loc=None, ip=None):
    return maybe_cast(
        get_result_or_results(MakeRangeOp(result, start, end, loc=loc, ip=ip))
    )


def make_tensor_ptr(
    result: Type,
    base: Value,
    shape: list[Value],
    strides: list[Value],
    offsets: list[Value],
    order,
    *,
    loc=None,
    ip=None,
):
    return maybe_cast(
        get_result_or_results(
            MakeTensorPtrOp(
                result, base, shape, strides, offsets, order, loc=loc, ip=ip
            )
        )
    )


def print(prefix, args: list[Value], *, loc=None, ip=None):
    return maybe_cast(get_result_or_results(PrintOp(prefix, args, loc=loc, ip=ip)))


def ptr_to_int(result: Type, from_: Value, *, loc=None, ip=None):
    return maybe_cast(get_result_or_results(PtrToIntOp(result, from_, loc=loc, ip=ip)))


@region_op
def reduce(result: list[Type], operands_: list[Value], axis, *, loc=None, ip=None):
    return ReduceOp(result, operands_, axis, loc=loc, ip=ip)


def return_(result: list[Value], *, loc=None, ip=None):
    return maybe_cast(get_result_or_results(ReduceReturnOp(result, loc=loc, ip=ip)))


def return_(operands_: list[Value], *, loc=None, ip=None):
    return maybe_cast(get_result_or_results(ReturnOp(operands_, loc=loc, ip=ip)))


def splat(result: Type, src: Value, *, loc=None, ip=None):
    return maybe_cast(get_result_or_results(SplatOp(result, src, loc=loc, ip=ip)))


def store(
    ptr: Value,
    value: Value,
    *,
    mask=None,
    boundary_check=None,
    cache=None,
    evict=None,
    loc=None,
    ip=None,
):
    return maybe_cast(
        get_result_or_results(
            StoreOp(
                ptr,
                value,
                mask=mask,
                boundaryCheck=boundary_check,
                cache=cache,
                evict=evict,
                loc=loc,
                ip=ip,
            )
        )
    )


def trans(src: Value, *, loc=None, ip=None):
    return maybe_cast(get_result_or_results(TransOp(src, loc=loc, ip=ip)))


def view(result: Type, src: Value, *, loc=None, ip=None):
    return maybe_cast(get_result_or_results(ViewOp(result, src, loc=loc, ip=ip)))


__all__ = [
    "addptr",
    "advance",
    "assert_",
    "atomic_cas",
    "atomic_rmw",
    "bitcast",
    "broadcast",
    "call",
    "cat",
    "dot",
    "expand_dims",
    "ext_elemwise",
    "fp_to_fp",
    "func",
    "get_num_programs",
    "get_program_id",
    "int_to_ptr",
    "load",
    "make_range",
    "make_tensor_ptr",
    "print",
    "ptr_to_int",
    "reduce",
    "return_",
    "return_",
    "splat",
    "store",
    "trans",
    "view",
]
