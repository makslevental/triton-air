from mlir_utils.dialects.ext.arith import Scalar
from mlir_utils.dialects.ext.func import FuncBase
from mlir_utils.dialects.ext.tensor import Tensor
from mlir_utils.types import i32_t, tensor_t
from mlir_utils.util import (
    make_maybe_no_args_decorator,
    get_user_code_loc,
    register_value_caster,
)
from triton_mlir_bindings._mlir_libs._mlir.ir import IntegerType
from triton_mlir_bindings.dialects.linalg.opdsl.lang.emitter import _is_integer_type
from triton_mlir_bindings.dialects.triton import FuncOp, ReturnOp, CallOp
from triton_mlir_bindings.ir import (
    Attribute,
    IntegerAttr,
    Value,
    RankedTensorType,
    register_attribute_builder,
    Context,
)

from mlir_utils.dialects import triton, arith
from triton_air.types import get_ptr_type, is_ptr_t


@register_value_caster(RankedTensorType.static_typeid, priority=0)
class TritonTensor(Tensor):
    def __add__(self, other: Tensor | Value, *, loc=None):
        if is_ptr_t(self.dtype):
            return addptr(self, other)

        if is_ptr_t(other.type) or _is_integer_type(other.type):
            assert self.has_static_shape()
            other = splat(other, self.shape)
            if is_ptr_t(other.dtype):
                self, other = other, self
            return addptr(self, other)

        if loc is None:
            loc = get_user_code_loc()
        return Tensor.__add__(self, other, loc=loc)

    def __lt__(self, other: Tensor | Value, *, loc=None):
        if is_ptr_t(other.type) or _is_integer_type(other.type):
            assert self.has_static_shape()
            other = splat(other, self.shape)
        if loc is None:
            loc = get_user_code_loc()
        return Tensor.__lt__(self, other, loc=loc)

    def __radd__(self, other):
        return self + other

    def __getitem__(self, mask):
        return load(self, mask)

    def __setitem__(self, mask, value, *, loc=None):
        if loc is None:
            loc = get_user_code_loc()
        triton.store(self, value, mask=mask, loc=loc)


@register_value_caster(IntegerType.static_typeid, priority=0)
class TritonScalar(Scalar):
    def __add__(self, other: TritonTensor | Value):
        if _is_integer_type(self.dtype) and isinstance(other, TritonTensor):
            return splat(self, other.shape) + other

        return Scalar.__add__(self, other)

    def __radd__(self, other):
        return self + other


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


@make_maybe_no_args_decorator
def jit(
    f,
    *,
    sym_visibility=None,
    arg_attrs=None,
    res_attrs=None,
    loc=None,
    ip=None,
):
    if loc is None:
        loc = get_user_code_loc()
    return FuncBase(
        body_builder=f,
        func_op_ctor=FuncOp.__base__,
        return_op_ctor=ReturnOp,
        call_op_ctor=CallOp,
        sym_visibility=sym_visibility,
        arg_attrs=arg_attrs,
        res_attrs=res_attrs,
        loc=loc,
        ip=ip,
    )


def arange(start, end, *, loc=None, ip=None):
    if loc is None:
        loc = get_user_code_loc()
    result_type = tensor_t(end - start, i32_t)
    return triton.make_range(result_type, start, end, loc=loc, ip=ip)


def splat(src: Value, sizes: tuple[int], *, loc=None, ip=None):
    if loc is None:
        loc = get_user_code_loc()
    result_type = tensor_t(*sizes, src.type)
    return triton.splat(result_type, src, loc=loc, ip=ip)


def addptr(ptr: Value, offset: Value, *, loc=None, ip=None):
    if loc is None:
        loc = get_user_code_loc()
    result_type = ptr.type
    return triton.addptr(result_type, ptr, offset, loc=loc, ip=ip)


def program_id(axis, *, loc=None, ip=None):
    if loc is None:
        loc = get_user_code_loc()
    return triton.get_program_id(axis, loc=loc, ip=ip)


def load(
    ptr: Value,
    mask: Value,
    cache="none",
    evict="normal",
    is_volatile=False,
    *,
    other=None,
    boundary_check=None,
    padding=None,
    loc=None,
    ip=None,
):
    if loc is None:
        loc = get_user_code_loc()
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
    if loc is None:
        loc = get_user_code_loc()
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


def cdiv(lhs, rhs, *, loc=None, ip=None):
    if not isinstance(lhs, TritonScalar):
        lhs = TritonScalar(lhs, dtype=i32_t)
    if not isinstance(rhs, TritonScalar):
        rhs = TritonScalar(rhs, dtype=i32_t)

    return lhs / rhs
