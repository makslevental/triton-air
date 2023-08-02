from typing import Optional, Sequence

from mlir_utils.dialects import triton
from mlir_utils.dialects.ext.arith import Scalar, constant
from mlir_utils.dialects.ext.func import FuncBase
from mlir_utils.dialects.ext.tensor import Tensor, expand_dims
from mlir_utils.types import i32_t, tensor_t, f32_t
from mlir_utils.util import (
    make_maybe_no_args_decorator,
    get_user_code_loc,
    register_value_caster,
)
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
    IntegerType,
    Type,
)

from triton_air.types import get_ptr_type, is_ptr_t, p_f16_t


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
    return TritonTensor(triton.make_range(result_type, start, end, loc=loc, ip=ip))


def splat(src: Value, sizes: tuple[int], *, loc=None, ip=None):
    if loc is None:
        loc = get_user_code_loc()
    result_type = tensor_t(*sizes, src.type)
    return TritonTensor(triton.splat(result_type, src, loc=loc, ip=ip))


def zeros(shape: Sequence[int], dtype: Optional[Type] = f32_t):
    return TritonTensor(constant(0, RankedTensorType.get(shape, dtype)))


def broadcast(shape: list[int], src: Tensor, *, loc=None, ip=None):
    if loc is None:
        loc = get_user_code_loc()
    return TritonTensor(
        triton.broadcast(RankedTensorType.get(shape, src.dtype), src, loc=loc, ip=ip)
    )


def addptr(ptr: "TritonTensor", offset: Value | int, *, loc=None, ip=None):
    if loc is None:
        loc = get_user_code_loc()
    result_type = ptr.type
    if isinstance(offset, int):
        offset = constant(offset, type=get_ptr_type(ptr.type))
    if isinstance(offset, Scalar):
        assert ptr.has_static_shape()
        offset = splat(offset, ptr.shape)
    if ptr.shape != offset.shape:
        offset = broadcast(ptr.shape, offset, loc=loc, ip=ip)
    return TritonTensor(triton.addptr(result_type, ptr, offset, loc=loc, ip=ip))


def broadcast_binary(lhs: Tensor, rhs: Tensor) -> tuple[Tensor, Tensor]:
    lhs_shape = lhs.shape
    rhs_shape = rhs.shape

    if len(lhs_shape) < len(rhs_shape):
        # Add new axes to lhs
        for dim in range(len(lhs_shape), len(rhs_shape)):
            lhs = expand_dims(lhs.handle, [1] + lhs_shape)
            lhs_shape = lhs.shape
    elif len(rhs_shape) < len(lhs_shape):
        # Add new axes to rhs
        for dim in range(len(rhs_shape), len(lhs_shape)):
            rhs = expand_dims(rhs.handle, [1] + rhs_shape)
            rhs_shape = rhs.shape
    assert len(rhs_shape) == len(lhs_shape)

    ret_shape = []
    for i, left in enumerate(lhs_shape):
        right = rhs_shape[i]
        if left == 1:
            ret_shape.append(right)
        elif right == 1:
            ret_shape.append(left)
        elif left == right:
            ret_shape.append(left)
        else:
            raise ValueError(
                "Cannot make_shape_compatible: incompatible dimensions "
                "at index " + str(i) + ": " + str(left) + " and " + str(right)
            )
    if lhs_shape != ret_shape:
        lhs = broadcast(ret_shape, lhs)
    if rhs_shape != ret_shape:
        rhs = broadcast(ret_shape, rhs)
    return lhs, rhs


class TritonTensor(Tensor):
    def coerce(self, other) -> tuple["Tensor", "Tensor"]:
        if not (isinstance(other, TritonPointer) or isinstance(other, Tensor)):
            self, other = super().coerce(other)

        if isinstance(other, TritonPointer):
            assert self.has_static_shape()
            other = splat(other, self.shape)

        if isinstance(other, Tensor) and self.shape != other.shape:
            self, other = broadcast_binary(self, other)

        return self, other

    def __add__(self, other: Tensor | Value, *, loc=None):
        if loc is None:
            loc = get_user_code_loc()
        if is_ptr_t(self):
            return addptr(self, other, loc=loc)

        return TritonTensor(super().__add__(other))

    def __getitem__(self, mask):
        if is_ptr_t(self):
            return load(self, mask)
        return TritonTensor(super().__getitem__(mask))

    def __setitem__(self, mask, value, *, loc=None):
        if loc is None:
            loc = get_user_code_loc()
        triton.store(self, value, mask=mask, loc=loc)


# it doesn't matter which p_f* is used here, the typeid is the same for all
@register_value_caster(IntegerType.static_typeid, 0)
@register_value_caster(p_f16_t.typeid)
class TritonPointer(Scalar):
    def __add__(self, other: Scalar | Tensor, *, loc=None):
        if isinstance(other, Tensor):
            # broadcast and wrap in TritonTensor for the subclassed __add__
            other, self = map(TritonTensor, other.coerce(self))
            return self + other
        else:
            return super().__add__(other)

    def __mul__(self, other: Scalar | Tensor, *, loc=None):
        if isinstance(other, Tensor):
            # broadcast and wrap in TritonTensor for the subclassed __mul__
            other, self = map(TritonTensor, other.coerce(self))
            return self * other
        else:
            return super().__mul__(other)


@register_value_caster(RankedTensorType.static_typeid, 0)
def maybe_cast_triton_tensor(val: Value):
    if is_ptr_t(val):
        return TritonTensor(val)


def program_id(axis, *, loc=None, ip=None):
    if loc is None:
        loc = get_user_code_loc()
    return TritonPointer(triton.get_program_id(axis, loc=loc, ip=ip))


def num_programs(axis, *, loc=None, ip=None):
    if loc is None:
        loc = get_user_code_loc()
    return TritonPointer(triton.get_num_programs(axis, loc=loc, ip=ip))


def cdiv(lhs, rhs, *, loc=None, ip=None):
    return lhs / rhs


def load(
    ptr: TritonTensor,
    mask: TritonTensor,
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
    if isinstance(other, (int, float, bool)):
        other = constant(other, type=get_ptr_type(ptr.type))
    if other is not None:
        if isinstance(other, Scalar):
            other = splat(other, ptr.shape)
        if ptr.shape != other.shape:
            other = broadcast(ptr.shape, other, loc=loc, ip=ip)
    if ptr.shape != mask.shape:
        mask = broadcast(ptr.shape, mask, loc=loc, ip=ip)

    return TritonTensor(
        triton.load(
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
    )


def store(
    ptr: TritonTensor,
    value: TritonTensor,
    mask: TritonTensor,
    *,
    boundary_check=None,
    cache=None,
    evict=None,
    loc=None,
    ip=None,
):
    if loc is None:
        loc = get_user_code_loc()
    if ptr.shape != value.shape:
        ptr = broadcast(value.shape, ptr, loc=loc, ip=ip)
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


def dot(
    a: TritonTensor,
    b: TritonTensor,
    *,
    c: TritonTensor = None,
    allow_tf32=True,
    loc=None,
    ip=None,
):
    if loc is None:
        loc = get_user_code_loc()

    if c is None:
        c = constant(1.0, type=a.dtype)
    if isinstance(c, Scalar):
        assert a.has_static_shape()
        c = splat(c, a.shape)

    return triton.dot(
        a,
        b,
        c,
        allow_tf32,
        loc=loc,
        ip=ip,
    )
