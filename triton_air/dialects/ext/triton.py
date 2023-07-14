from functools import partial

from mlir_utils.dialects.ext.func import func_base
from mlir_utils.dialects.util import (
    make_maybe_no_args_decorator,
)
from mlir_utils.types import i32_t, tensor_t
from triton_mlir_bindings._mlir_libs._mlir.ir import IntegerType
from triton_mlir_bindings.dialects.triton import (
    FuncOp,
    ReturnOp,
    CallOp,
)
from triton_mlir_bindings.ir import (
    Attribute,
    IntegerAttr,
    Value,
    RankedTensorType,
    register_attribute_builder,
    Context,
)

from mlir_utils.dialects import triton
from triton_air.types import get_ptr_type

jit = make_maybe_no_args_decorator(
    partial(func_base, FuncOp=FuncOp.__base__, ReturnOp=ReturnOp, CallOp=CallOp)
)


def arange(start, end, *, loc=None, ip=None):
    result_type = tensor_t(end - start, i32_t)
    return triton.make_range(result_type, start, end, loc=loc, ip=ip)


def splat(src: Value, sizes: tuple[int], *, loc=None, ip=None):
    result_type = tensor_t(*sizes, src.type)
    return triton.splat(result_type, src, loc=loc, ip=ip)


def addptr(ptr: Value, offset: Value, *, loc=None, ip=None):
    result_type = ptr.type
    return triton.addptr(result_type, ptr, offset, loc=loc, ip=ip)


@register_attribute_builder("TT_CacheModifierAttr")
def _tT_CacheModifierAttr(cache_modifier: str | Attribute, context: Context):
    cache_modifiers = {
        "none": 1,
        "ca": 2,
        "g": 3,
    }
    if isinstance(cache_modifier, Attribute):
        return cache_modifier
    assert (
        cache_modifier in cache_modifiers
    ), f"cache_modifier {cache_modifier} not in cache_modifiers"
    return IntegerAttr.get(
        IntegerType.get_signless(32, context=context), cache_modifiers[cache_modifier]
    )


@register_attribute_builder("TT_EvictionPolicyAttr")
def _tT_EvictionPolicyAttr(eviction_policy: str | Attribute, context: Context):
    eviction_policies = {
        "normal": 1,
        "first": 2,
        "last": 3,
    }
    if isinstance(eviction_policy, Attribute):
        return eviction_policy
    assert (
        eviction_policy in eviction_policies
    ), f"eviction_policy {eviction_policy} not in eviction_policies"
    return IntegerAttr.get(
        IntegerType.get_signless(32, context=context),
        eviction_policies[eviction_policy],
    )


@register_attribute_builder("TT_ProgramDim")
def _tT_ProgramDim(dim: str | Attribute, context: Context):
    dims = {
        "x": 0,
        "y": 1,
        "z": 2,
    }
    if isinstance(dim, Attribute):
        return dim
    assert dim in dims, f"dim {dim} not in dims"
    return IntegerAttr.get(
        IntegerType.get_signless(32, context=context),
        dims[dim],
    )


@register_attribute_builder("TT_PaddingOptionAttr")
def _tT_PaddingOptionAttr(padding_option: str | Attribute, context: Context):
    padding_options = {
        "zero": 1,
        "nan": 2,
    }
    if isinstance(padding_option, Attribute):
        return padding_option
    assert (
        padding_option in padding_options
    ), f"padding_option {padding_option} not in padding_options"
    return IntegerAttr.get(
        IntegerType.get_signless(32, context=context),
        padding_options[padding_option],
    )


@register_attribute_builder("TT_AtomicRMWAttr")
def _tT_AtomicRMWAttr(rmwop: str | Attribute, context: Context):
    rmwops = {
        "and": 1,
        "or": 2,
        "xor": 3,
        "add": 4,
        "fadd": 5,
        "max": 6,
        "min": 7,
        "umax": 8,
        "umin": 9,
        "exch": 10,
    }
    if isinstance(rmwop, Attribute):
        return rmwop
    assert rmwop in rmwops, f"rmwop {rmwop} not in rmwops"
    return IntegerAttr.get(
        IntegerType.get_signless(32, context=context),
        rmwops[rmwop],
    )


def load(
    ptr: Value,
    mask: Value,
    cache,
    evict,
    is_volatile,
    *,
    other=None,
    boundary_check=None,
    padding=None,
    loc=None,
    ip=None,
):
    if not RankedTensorType.isinstance(ptr.type):
        raise ValueError(f"{ptr=} must be RankedTensorType")
    ptr_type = RankedTensorType(ptr.type)
    if ptr_type.has_static_shape:
        result_type = RankedTensorType.get(ptr_type.shape, get_ptr_type(ptr.type))
    else:
        raise ValueError(f"dynamic shape for {ptr=} not supported")
    return triton.load(
        result_type,
        ptr,
        cache,
        evict,
        is_volatile,
        mask=mask,
        other=other,
        boundary_check=boundary_check,
        padding=padding,
        loc=loc,
        ip=ip,
    )


def store(
    ptr: Value,
    value: Value,
    mask: Value,
    *,
    boundary_check=None,
    cache=None,
    evict=None,
    loc=None,
    ip=None,
):
    return triton.store(
        ptr,
        value,
        mask=mask,
        boundary_check=boundary_check,
        cache=cache,
        evict=evict,
        loc=loc,
        ip=ip,
    )
