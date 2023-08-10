from mlir_utils.dialects.util import get_result_or_results, maybe_cast
from triton_mlir_bindings.ir import Value, Type

from ._air_transform_ops_gen import (
    CopyToDmaOp,
    FuseIntoContainingMemrefOp,
    GetSegmentForOp,
    LinalgPromoteOp,
    LinalgTileOp,
    ParToHerdOp,
    ParToLaunchOp,
    PipelineReduceOp,
    SegmentToAIEOp,
)


def copy_to_dma(result: Type, target: Value, *, loc=None, ip=None):
    return maybe_cast(
        get_result_or_results(CopyToDmaOp(result, target, loc=loc, ip=ip))
    )


def fuse_into_containing_op(
    fused_op: Type, producer_op: Value, containing_op: Value, *, loc=None, ip=None
):
    return maybe_cast(
        get_result_or_results(
            FuseIntoContainingMemrefOp(
                fused_op, producer_op, containing_op, loc=loc, ip=ip
            )
        )
    )


def get_segment_for(parent: Type, target: Value, *, loc=None, ip=None):
    return maybe_cast(
        get_result_or_results(GetSegmentForOp(parent, target, loc=loc, ip=ip))
    )


def linalg_promote(
    transformed: Type,
    target: Value,
    *,
    operands_to_promote=None,
    group_size=None,
    use_full_tile_buffers=None,
    use_full_tiles_by_default=None,
    use_alloca=None,
    alignment=None,
    memory_space=None,
    loc=None,
    ip=None,
):
    return maybe_cast(
        get_result_or_results(
            LinalgPromoteOp(
                transformed,
                target,
                operands_to_promote=operands_to_promote,
                group_size=group_size,
                use_full_tile_buffers=use_full_tile_buffers,
                use_full_tiles_by_default=use_full_tiles_by_default,
                use_alloca=use_alloca,
                alignment=alignment,
                memory_space=memory_space,
                loc=loc,
                ip=ip,
            )
        )
    )


def linalg_tile(
    tiled_linalg_op: Type,
    loops: list[Type],
    target: Value,
    dynamic_sizes: list[Value],
    *,
    static_sizes=None,
    interchange=None,
    loc=None,
    ip=None,
):
    return maybe_cast(
        get_result_or_results(
            LinalgTileOp(
                tiled_linalg_op,
                loops,
                target,
                dynamic_sizes,
                static_sizes=static_sizes,
                interchange=interchange,
                loc=loc,
                ip=ip,
            )
        )
    )


def par_to_herd(result: Type, target: Value, *, loc=None, ip=None):
    return maybe_cast(
        get_result_or_results(ParToHerdOp(result, target, loc=loc, ip=ip))
    )


def par_to_launch(result: Type, target: Value, *, loc=None, ip=None):
    return maybe_cast(
        get_result_or_results(ParToLaunchOp(result, target, loc=loc, ip=ip))
    )


def pipeline_reduce(
    result: Type,
    target: Value,
    *,
    tile_size=None,
    pipeline_depth=None,
    direction=None,
    promote=None,
    loc=None,
    ip=None,
):
    return maybe_cast(
        get_result_or_results(
            PipelineReduceOp(
                result,
                target,
                tile_size=tile_size,
                pipeline_depth=pipeline_depth,
                direction=direction,
                promote=promote,
                loc=loc,
                ip=ip,
            )
        )
    )


def segment_to_aie(transformed: Type, target: Value, *, loc=None, ip=None):
    return maybe_cast(
        get_result_or_results(SegmentToAIEOp(transformed, target, loc=loc, ip=ip))
    )


__all__ = [
    "copy_to_dma",
    "fuse_into_containing_op",
    "get_segment_for",
    "linalg_promote",
    "linalg_tile",
    "par_to_herd",
    "par_to_launch",
    "pipeline_reduce",
    "segment_to_aie",
]
