from triton_mlir_bindings.ir import Value, Attribute, Type
from triton_pp.dialects._air_ops_gen import (
    AllocOp,
    ChannelGetOp,
    ChannelOp,
    ChannelPutOp,
    DeallocOp,
    DmaMemcpyNdOp,
    ExecuteOp,
    ExecuteTerminatorOp,
    HerdPipelineOp,
    HerdTerminatorOp,
    LaunchTerminatorOp,
    PipelineGetOp,
    PipelinePutOp,
    PipelineStageOp,
    PipelineTerminatorOp,
    PipelineYieldOp,
    SegmentTerminatorOp,
    WaitAllOp,
)
from mlir_utils.util import (
    get_result_or_results,
    maybe_cast,
    region_op,
    get_user_code_loc,
)


def alloc(
    async_token: Type,
    result: Type,
    async_dependencies: list[Value],
    *,
    loc=None,
    ip=None
):
    if loc is None:
        loc = get_user_code_loc()
    return maybe_cast(
        get_result_or_results(
            AllocOp(async_token, result, async_dependencies, loc=loc, ip=ip)
        )
    )


def channel_get(
    async_token: Type,
    async_dependencies: list[Value],
    chan_name,
    indices: list[Value],
    dst: Value,
    dst_offsets: list[Value],
    dst_sizes: list[Value],
    dst_strides: list[Value],
    *,
    loc=None,
    ip=None
):
    if loc is None:
        loc = get_user_code_loc()
    return maybe_cast(
        get_result_or_results(
            ChannelGetOp(
                async_token,
                async_dependencies,
                chan_name,
                indices,
                dst,
                dst_offsets,
                dst_sizes,
                dst_strides,
                loc=loc,
                ip=ip,
            )
        )
    )


def channel(sym_name, *, size=None, loc=None, ip=None):
    if loc is None:
        loc = get_user_code_loc()
    return maybe_cast(
        get_result_or_results(ChannelOp(sym_name, size=size, loc=loc, ip=ip))
    )


def channel_put(
    async_token: Type,
    async_dependencies: list[Value],
    chan_name,
    indices: list[Value],
    src: Value,
    src_offsets: list[Value],
    src_sizes: list[Value],
    src_strides: list[Value],
    *,
    loc=None,
    ip=None
):
    if loc is None:
        loc = get_user_code_loc()
    return maybe_cast(
        get_result_or_results(
            ChannelPutOp(
                async_token,
                async_dependencies,
                chan_name,
                indices,
                src,
                src_offsets,
                src_sizes,
                src_strides,
                loc=loc,
                ip=ip,
            )
        )
    )


def dealloc(
    async_token: Type,
    async_dependencies: list[Value],
    memref: Value,
    *,
    loc=None,
    ip=None
):
    if loc is None:
        loc = get_user_code_loc()
    return maybe_cast(
        get_result_or_results(
            DeallocOp(async_token, async_dependencies, memref, loc=loc, ip=ip)
        )
    )


def dma_memcpy_nd(
    async_token: Type,
    async_dependencies: list[Value],
    dst: Value,
    dst_offsets: list[Value],
    dst_sizes: list[Value],
    dst_strides: list[Value],
    src: Value,
    src_offsets: list[Value],
    src_sizes: list[Value],
    src_strides: list[Value],
    *,
    loc=None,
    ip=None
):
    if loc is None:
        loc = get_user_code_loc()
    return maybe_cast(
        get_result_or_results(
            DmaMemcpyNdOp(
                async_token,
                async_dependencies,
                dst,
                dst_offsets,
                dst_sizes,
                dst_strides,
                src,
                src_offsets,
                src_sizes,
                src_strides,
                loc=loc,
                ip=ip,
            )
        )
    )


@region_op
def execute(
    async_token: Type,
    results_: list[Type],
    async_dependencies: list[Value],
    *,
    loc=None,
    ip=None
):
    if loc is None:
        loc = get_user_code_loc()
    return ExecuteOp(async_token, results_, async_dependencies, loc=loc, ip=ip)


def execute_terminator(results_: list[Value] = None, *, loc=None, ip=None):
    if loc is None:
        loc = get_user_code_loc()
    results_ = results_ or []
    return maybe_cast(
        get_result_or_results(ExecuteTerminatorOp(results_, loc=loc, ip=ip))
    )


@region_op
def pipeline(*, loc=None, ip=None):
    if loc is None:
        loc = get_user_code_loc()
    return HerdPipelineOp(loc=loc, ip=ip)


def herd_terminator(*, loc=None, ip=None):
    if loc is None:
        loc = get_user_code_loc()
    return maybe_cast(get_result_or_results(HerdTerminatorOp(loc=loc, ip=ip)))


def launch_terminator(*, loc=None, ip=None):
    if loc is None:
        loc = get_user_code_loc()
    return maybe_cast(get_result_or_results(LaunchTerminatorOp(loc=loc, ip=ip)))


def pipeline_get(results_: list[Type], src0: Value, src1: Value, *, loc=None, ip=None):
    if loc is None:
        loc = get_user_code_loc()
    return maybe_cast(
        get_result_or_results(PipelineGetOp(results_, src0, src1, loc=loc, ip=ip))
    )


def pipeline_put(dst0: Value, dst1: Value, opers: list[Value], *, loc=None, ip=None):
    if loc is None:
        loc = get_user_code_loc()
    return maybe_cast(
        get_result_or_results(PipelinePutOp(dst0, dst1, opers, loc=loc, ip=ip))
    )


@region_op
def pipeline_stage(results_: list[Type], opers: list[Value], *, loc=None, ip=None):
    if loc is None:
        loc = get_user_code_loc()
    return PipelineStageOp(results_, opers, loc=loc, ip=ip)


def pipeline_terminator(opers: list[Value], *, loc=None, ip=None):
    if loc is None:
        loc = get_user_code_loc()
    return maybe_cast(
        get_result_or_results(PipelineTerminatorOp(opers, loc=loc, ip=ip))
    )


def pipeline_yield(opers: list[Value], *, loc=None, ip=None):
    if loc is None:
        loc = get_user_code_loc()
    return maybe_cast(get_result_or_results(PipelineYieldOp(opers, loc=loc, ip=ip)))


def segment_terminator(*, loc=None, ip=None):
    if loc is None:
        loc = get_user_code_loc()
    return maybe_cast(get_result_or_results(SegmentTerminatorOp(loc=loc, ip=ip)))


def wait_all(async_token: Type, async_dependencies: list[Value], *, loc=None, ip=None):
    if loc is None:
        loc = get_user_code_loc()
    return maybe_cast(
        get_result_or_results(
            WaitAllOp(async_token, async_dependencies, loc=loc, ip=ip)
        )
    )


__all__ = [
    "alloc",
    "channel_get",
    "channel",
    "channel_put",
    "dealloc",
    "dma_memcpy_nd",
    "execute",
    "execute_terminator",
    "pipeline",
    "herd_terminator",
    "launch_terminator",
    "pipeline_get",
    "pipeline_put",
    "pipeline_stage",
    "pipeline_terminator",
    "pipeline_yield",
    "segment_terminator",
    "wait_all",
]
