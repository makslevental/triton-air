from triton_mlir_bindings.dialects._ods_common import _cext as _ods_cext
from triton_mlir_bindings.dialects._ods_common import extend_opview_class as _ods_extend_opview_class, segmented_accessor as _ods_segmented_accessor, equally_sized_accessor as _ods_equally_sized_accessor, get_default_loc_context as _ods_get_default_loc_context, get_op_result_or_value as _get_op_result_or_value, get_op_results_or_values as _get_op_results_or_values
_ods_ir = _ods_cext.ir

try:
  from . import _air_ops_ext as _ods_ext_module
except ImportError:
  _ods_ext_module = None

import builtins


@_ods_cext.register_dialect
class _Dialect(_ods_ir.Dialect):
  DIALECT_NAMESPACE = "air"
  pass


@_ods_cext.register_operation(_Dialect)
@_ods_extend_opview_class(_ods_ext_module)
class AllocOp(_ods_ir.OpView):
  OPERATION_NAME = "air.alloc"

  _ODS_REGIONS = (0, True)

  def __init__(self, async_token, result, async_dependencies, *, loc=None, ip=None):
    operands = []
    results = []
    attributes = {}
    regions = None
    operands.extend(_get_op_results_or_values(async_dependencies))
    _ods_context = _ods_get_default_loc_context(loc)
    if async_token is not None: results.append(async_token)
    results.append(result)
    _ods_successors = None
    super().__init__(self.build_generic(
      attributes=attributes, results=results, operands=operands,
      successors=_ods_successors, regions=regions, loc=loc, ip=ip))

  @builtins.property
  def async_dependencies(self):
    _ods_variadic_group_length = len(self.operation.operands) - 1 + 1
    return self.operation.operands[0:0 + _ods_variadic_group_length]

  @builtins.property
  def async_token(self):
    return None if len(self.operation.results) < 2 else self.operation.results[0]

  @builtins.property
  def result(self):
    _ods_variadic_group_length = len(self.operation.results) - 2 + 1
    return self.operation.results[1 + _ods_variadic_group_length - 1]

@_ods_cext.register_operation(_Dialect)
@_ods_extend_opview_class(_ods_ext_module)
class ChannelGetOp(_ods_ir.OpView):
  OPERATION_NAME = "air.channel.get"

  _ODS_OPERAND_SEGMENTS = [-1,-1,1,-1,-1,-1,]

  _ODS_REGIONS = (0, True)

  def __init__(self, async_token, async_dependencies, chan_name, indices, dst, dst_offsets, dst_sizes, dst_strides, *, loc=None, ip=None):
    operands = []
    results = []
    attributes = {}
    regions = None
    operands.append(_get_op_results_or_values(async_dependencies))
    operands.append(_get_op_results_or_values(indices))
    operands.append(_get_op_result_or_value(dst))
    operands.append(_get_op_results_or_values(dst_offsets))
    operands.append(_get_op_results_or_values(dst_sizes))
    operands.append(_get_op_results_or_values(dst_strides))
    _ods_context = _ods_get_default_loc_context(loc)
    attributes["chan_name"] = (chan_name if (
    issubclass(type(chan_name), _ods_ir.Attribute) or
    not _ods_ir.AttrBuilder.contains('FlatSymbolRefAttr')) else
      _ods_ir.AttrBuilder.get('FlatSymbolRefAttr')(chan_name, context=_ods_context))
    if async_token is not None: results.append(async_token)
    _ods_successors = None
    super().__init__(self.build_generic(
      attributes=attributes, results=results, operands=operands,
      successors=_ods_successors, regions=regions, loc=loc, ip=ip))

  @builtins.property
  def async_dependencies(self):
    operand_range = _ods_segmented_accessor(
         self.operation.operands,
         self.operation.attributes["operand_segment_sizes"], 0)
    return operand_range

  @builtins.property
  def indices(self):
    operand_range = _ods_segmented_accessor(
         self.operation.operands,
         self.operation.attributes["operand_segment_sizes"], 1)
    return operand_range

  @builtins.property
  def dst(self):
    operand_range = _ods_segmented_accessor(
         self.operation.operands,
         self.operation.attributes["operand_segment_sizes"], 2)
    return operand_range[0]

  @builtins.property
  def dst_offsets(self):
    operand_range = _ods_segmented_accessor(
         self.operation.operands,
         self.operation.attributes["operand_segment_sizes"], 3)
    return operand_range

  @builtins.property
  def dst_sizes(self):
    operand_range = _ods_segmented_accessor(
         self.operation.operands,
         self.operation.attributes["operand_segment_sizes"], 4)
    return operand_range

  @builtins.property
  def dst_strides(self):
    operand_range = _ods_segmented_accessor(
         self.operation.operands,
         self.operation.attributes["operand_segment_sizes"], 5)
    return operand_range

  @builtins.property
  def async_token(self):
    return None if len(self.operation.results) < 1 else self.operation.results[0]

@_ods_cext.register_operation(_Dialect)
@_ods_extend_opview_class(_ods_ext_module)
class ChannelOp(_ods_ir.OpView):
  OPERATION_NAME = "air.channel"

  _ODS_REGIONS = (0, True)

  def __init__(self, sym_name, *, size=None, loc=None, ip=None):
    operands = []
    results = []
    attributes = {}
    regions = None
    _ods_context = _ods_get_default_loc_context(loc)
    attributes["sym_name"] = (sym_name if (
    issubclass(type(sym_name), _ods_ir.Attribute) or
    not _ods_ir.AttrBuilder.contains('SymbolNameAttr')) else
      _ods_ir.AttrBuilder.get('SymbolNameAttr')(sym_name, context=_ods_context))
    if size is not None: attributes["size"] = (size if (
        issubclass(type(size), _ods_ir.Attribute) or
        not _ods_ir.AttrBuilder.contains('I64ArrayAttr')) else
          _ods_ir.AttrBuilder.get('I64ArrayAttr')(size, context=_ods_context))
    _ods_successors = None
    super().__init__(self.build_generic(
      attributes=attributes, results=results, operands=operands,
      successors=_ods_successors, regions=regions, loc=loc, ip=ip))

  @builtins.property
  def sym_name(self):
    return _ods_ir.StringAttr(self.operation.attributes["sym_name"])

  @sym_name.setter
  def sym_name(self, value):
    if value is None:
      raise ValueError("'None' not allowed as value for mandatory attributes")
    self.operation.attributes["sym_name"] = value

@_ods_cext.register_operation(_Dialect)
@_ods_extend_opview_class(_ods_ext_module)
class ChannelPutOp(_ods_ir.OpView):
  OPERATION_NAME = "air.channel.put"

  _ODS_OPERAND_SEGMENTS = [-1,-1,1,-1,-1,-1,]

  _ODS_REGIONS = (0, True)

  def __init__(self, async_token, async_dependencies, chan_name, indices, src, src_offsets, src_sizes, src_strides, *, loc=None, ip=None):
    operands = []
    results = []
    attributes = {}
    regions = None
    operands.append(_get_op_results_or_values(async_dependencies))
    operands.append(_get_op_results_or_values(indices))
    operands.append(_get_op_result_or_value(src))
    operands.append(_get_op_results_or_values(src_offsets))
    operands.append(_get_op_results_or_values(src_sizes))
    operands.append(_get_op_results_or_values(src_strides))
    _ods_context = _ods_get_default_loc_context(loc)
    attributes["chan_name"] = (chan_name if (
    issubclass(type(chan_name), _ods_ir.Attribute) or
    not _ods_ir.AttrBuilder.contains('FlatSymbolRefAttr')) else
      _ods_ir.AttrBuilder.get('FlatSymbolRefAttr')(chan_name, context=_ods_context))
    if async_token is not None: results.append(async_token)
    _ods_successors = None
    super().__init__(self.build_generic(
      attributes=attributes, results=results, operands=operands,
      successors=_ods_successors, regions=regions, loc=loc, ip=ip))

  @builtins.property
  def async_dependencies(self):
    operand_range = _ods_segmented_accessor(
         self.operation.operands,
         self.operation.attributes["operand_segment_sizes"], 0)
    return operand_range

  @builtins.property
  def indices(self):
    operand_range = _ods_segmented_accessor(
         self.operation.operands,
         self.operation.attributes["operand_segment_sizes"], 1)
    return operand_range

  @builtins.property
  def src(self):
    operand_range = _ods_segmented_accessor(
         self.operation.operands,
         self.operation.attributes["operand_segment_sizes"], 2)
    return operand_range[0]

  @builtins.property
  def src_offsets(self):
    operand_range = _ods_segmented_accessor(
         self.operation.operands,
         self.operation.attributes["operand_segment_sizes"], 3)
    return operand_range

  @builtins.property
  def src_sizes(self):
    operand_range = _ods_segmented_accessor(
         self.operation.operands,
         self.operation.attributes["operand_segment_sizes"], 4)
    return operand_range

  @builtins.property
  def src_strides(self):
    operand_range = _ods_segmented_accessor(
         self.operation.operands,
         self.operation.attributes["operand_segment_sizes"], 5)
    return operand_range

  @builtins.property
  def async_token(self):
    return None if len(self.operation.results) < 1 else self.operation.results[0]

@_ods_cext.register_operation(_Dialect)
@_ods_extend_opview_class(_ods_ext_module)
class CustomOp(_ods_ir.OpView):
  OPERATION_NAME = "air.custom"

  _ODS_OPERAND_SEGMENTS = [-1,-1,]

  _ODS_REGIONS = (0, True)

  @builtins.property
  def async_dependencies(self):
    operand_range = _ods_segmented_accessor(
         self.operation.operands,
         self.operation.attributes["operand_segment_sizes"], 0)
    return operand_range

  @builtins.property
  def custom_operands(self):
    operand_range = _ods_segmented_accessor(
         self.operation.operands,
         self.operation.attributes["operand_segment_sizes"], 1)
    return operand_range

  @builtins.property
  def async_token(self):
    return None if len(self.operation.results) < 1 else self.operation.results[0]

@_ods_cext.register_operation(_Dialect)
@_ods_extend_opview_class(_ods_ext_module)
class DeallocOp(_ods_ir.OpView):
  OPERATION_NAME = "air.dealloc"

  _ODS_REGIONS = (0, True)

  def __init__(self, async_token, async_dependencies, memref, *, loc=None, ip=None):
    operands = []
    results = []
    attributes = {}
    regions = None
    operands.extend(_get_op_results_or_values(async_dependencies))
    operands.append(_get_op_result_or_value(memref))
    _ods_context = _ods_get_default_loc_context(loc)
    if async_token is not None: results.append(async_token)
    _ods_successors = None
    super().__init__(self.build_generic(
      attributes=attributes, results=results, operands=operands,
      successors=_ods_successors, regions=regions, loc=loc, ip=ip))

  @builtins.property
  def async_dependencies(self):
    _ods_variadic_group_length = len(self.operation.operands) - 2 + 1
    return self.operation.operands[0:0 + _ods_variadic_group_length]

  @builtins.property
  def memref(self):
    _ods_variadic_group_length = len(self.operation.operands) - 2 + 1
    return self.operation.operands[1 + _ods_variadic_group_length - 1]

  @builtins.property
  def async_token(self):
    return None if len(self.operation.results) < 1 else self.operation.results[0]

@_ods_cext.register_operation(_Dialect)
@_ods_extend_opview_class(_ods_ext_module)
class DmaMemcpyNdOp(_ods_ir.OpView):
  OPERATION_NAME = "air.dma_memcpy_nd"

  _ODS_OPERAND_SEGMENTS = [-1,1,-1,-1,-1,1,-1,-1,-1,]

  _ODS_REGIONS = (0, True)

  def __init__(self, async_token, async_dependencies, dst, dst_offsets, dst_sizes, dst_strides, src, src_offsets, src_sizes, src_strides, *, loc=None, ip=None):
    operands = []
    results = []
    attributes = {}
    regions = None
    operands.append(_get_op_results_or_values(async_dependencies))
    operands.append(_get_op_result_or_value(dst))
    operands.append(_get_op_results_or_values(dst_offsets))
    operands.append(_get_op_results_or_values(dst_sizes))
    operands.append(_get_op_results_or_values(dst_strides))
    operands.append(_get_op_result_or_value(src))
    operands.append(_get_op_results_or_values(src_offsets))
    operands.append(_get_op_results_or_values(src_sizes))
    operands.append(_get_op_results_or_values(src_strides))
    _ods_context = _ods_get_default_loc_context(loc)
    if async_token is not None: results.append(async_token)
    _ods_successors = None
    super().__init__(self.build_generic(
      attributes=attributes, results=results, operands=operands,
      successors=_ods_successors, regions=regions, loc=loc, ip=ip))

  @builtins.property
  def async_dependencies(self):
    operand_range = _ods_segmented_accessor(
         self.operation.operands,
         self.operation.attributes["operand_segment_sizes"], 0)
    return operand_range

  @builtins.property
  def dst(self):
    operand_range = _ods_segmented_accessor(
         self.operation.operands,
         self.operation.attributes["operand_segment_sizes"], 1)
    return operand_range[0]

  @builtins.property
  def dst_offsets(self):
    operand_range = _ods_segmented_accessor(
         self.operation.operands,
         self.operation.attributes["operand_segment_sizes"], 2)
    return operand_range

  @builtins.property
  def dst_sizes(self):
    operand_range = _ods_segmented_accessor(
         self.operation.operands,
         self.operation.attributes["operand_segment_sizes"], 3)
    return operand_range

  @builtins.property
  def dst_strides(self):
    operand_range = _ods_segmented_accessor(
         self.operation.operands,
         self.operation.attributes["operand_segment_sizes"], 4)
    return operand_range

  @builtins.property
  def src(self):
    operand_range = _ods_segmented_accessor(
         self.operation.operands,
         self.operation.attributes["operand_segment_sizes"], 5)
    return operand_range[0]

  @builtins.property
  def src_offsets(self):
    operand_range = _ods_segmented_accessor(
         self.operation.operands,
         self.operation.attributes["operand_segment_sizes"], 6)
    return operand_range

  @builtins.property
  def src_sizes(self):
    operand_range = _ods_segmented_accessor(
         self.operation.operands,
         self.operation.attributes["operand_segment_sizes"], 7)
    return operand_range

  @builtins.property
  def src_strides(self):
    operand_range = _ods_segmented_accessor(
         self.operation.operands,
         self.operation.attributes["operand_segment_sizes"], 8)
    return operand_range

  @builtins.property
  def async_token(self):
    return None if len(self.operation.results) < 1 else self.operation.results[0]

@_ods_cext.register_operation(_Dialect)
@_ods_extend_opview_class(_ods_ext_module)
class ExecuteOp(_ods_ir.OpView):
  OPERATION_NAME = "air.execute"

  _ODS_REGIONS = (1, True)

  def __init__(self, async_token, results_, async_dependencies, *, loc=None, ip=None):
    operands = []
    results = []
    attributes = {}
    regions = None
    operands.extend(_get_op_results_or_values(async_dependencies))
    _ods_context = _ods_get_default_loc_context(loc)
    results.append(async_token)
    results.extend(results_)
    _ods_successors = None
    super().__init__(self.build_generic(
      attributes=attributes, results=results, operands=operands,
      successors=_ods_successors, regions=regions, loc=loc, ip=ip))

  @builtins.property
  def async_dependencies(self):
    _ods_variadic_group_length = len(self.operation.operands) - 1 + 1
    return self.operation.operands[0:0 + _ods_variadic_group_length]

  @builtins.property
  def async_token(self):
    return self.operation.results[0]

  @builtins.property
  def results_(self):
    _ods_variadic_group_length = len(self.operation.results) - 2 + 1
    return self.operation.results[1:1 + _ods_variadic_group_length]

  @builtins.property
  def body(self):
    return self.regions[0]

@_ods_cext.register_operation(_Dialect)
@_ods_extend_opview_class(_ods_ext_module)
class ExecuteTerminatorOp(_ods_ir.OpView):
  OPERATION_NAME = "air.execute_terminator"

  _ODS_REGIONS = (0, True)

  def __init__(self, results_, *, loc=None, ip=None):
    operands = []
    results = []
    attributes = {}
    regions = None
    operands.extend(_get_op_results_or_values(results_))
    _ods_context = _ods_get_default_loc_context(loc)
    _ods_successors = None
    super().__init__(self.build_generic(
      attributes=attributes, results=results, operands=operands,
      successors=_ods_successors, regions=regions, loc=loc, ip=ip))

  @builtins.property
  def results_(self):
    _ods_variadic_group_length = len(self.operation.operands) - 1 + 1
    return self.operation.operands[0:0 + _ods_variadic_group_length]

@_ods_cext.register_operation(_Dialect)
@_ods_extend_opview_class(_ods_ext_module)
class HerdOp(_ods_ir.OpView):
  OPERATION_NAME = "air.herd"

  _ODS_OPERAND_SEGMENTS = [-1,-1,-1,]

  _ODS_REGIONS = (1, True)

  @builtins.property
  def async_dependencies(self):
    operand_range = _ods_segmented_accessor(
         self.operation.operands,
         self.operation.attributes["operand_segment_sizes"], 0)
    return operand_range

  @builtins.property
  def sizes(self):
    operand_range = _ods_segmented_accessor(
         self.operation.operands,
         self.operation.attributes["operand_segment_sizes"], 1)
    return operand_range

  @builtins.property
  def herd_operands(self):
    operand_range = _ods_segmented_accessor(
         self.operation.operands,
         self.operation.attributes["operand_segment_sizes"], 2)
    return operand_range

  @builtins.property
  def async_token(self):
    return None if len(self.operation.results) < 1 else self.operation.results[0]

  @builtins.property
  def body(self):
    return self.regions[0]

@_ods_cext.register_operation(_Dialect)
@_ods_extend_opview_class(_ods_ext_module)
class HerdPipelineOp(_ods_ir.OpView):
  OPERATION_NAME = "air.pipeline"

  _ODS_REGIONS = (1, True)

  def __init__(self, *, loc=None, ip=None):
    operands = []
    results = []
    attributes = {}
    regions = None
    _ods_context = _ods_get_default_loc_context(loc)
    _ods_successors = None
    super().__init__(self.build_generic(
      attributes=attributes, results=results, operands=operands,
      successors=_ods_successors, regions=regions, loc=loc, ip=ip))

  @builtins.property
  def body(self):
    return self.regions[0]

@_ods_cext.register_operation(_Dialect)
@_ods_extend_opview_class(_ods_ext_module)
class HerdTerminatorOp(_ods_ir.OpView):
  OPERATION_NAME = "air.herd_terminator"

  _ODS_REGIONS = (0, True)

  def __init__(self, *, loc=None, ip=None):
    operands = []
    results = []
    attributes = {}
    regions = None
    _ods_context = _ods_get_default_loc_context(loc)
    _ods_successors = None
    super().__init__(self.build_generic(
      attributes=attributes, results=results, operands=operands,
      successors=_ods_successors, regions=regions, loc=loc, ip=ip))

@_ods_cext.register_operation(_Dialect)
@_ods_extend_opview_class(_ods_ext_module)
class LaunchOp(_ods_ir.OpView):
  OPERATION_NAME = "air.launch"

  _ODS_OPERAND_SEGMENTS = [-1,-1,-1,]

  _ODS_REGIONS = (1, True)

  @builtins.property
  def async_dependencies(self):
    operand_range = _ods_segmented_accessor(
         self.operation.operands,
         self.operation.attributes["operand_segment_sizes"], 0)
    return operand_range

  @builtins.property
  def sizes(self):
    operand_range = _ods_segmented_accessor(
         self.operation.operands,
         self.operation.attributes["operand_segment_sizes"], 1)
    return operand_range

  @builtins.property
  def launch_operands(self):
    operand_range = _ods_segmented_accessor(
         self.operation.operands,
         self.operation.attributes["operand_segment_sizes"], 2)
    return operand_range

  @builtins.property
  def async_token(self):
    return None if len(self.operation.results) < 1 else self.operation.results[0]

  @builtins.property
  def body(self):
    return self.regions[0]

@_ods_cext.register_operation(_Dialect)
@_ods_extend_opview_class(_ods_ext_module)
class LaunchTerminatorOp(_ods_ir.OpView):
  OPERATION_NAME = "air.launch_terminator"

  _ODS_REGIONS = (0, True)

  def __init__(self, *, loc=None, ip=None):
    operands = []
    results = []
    attributes = {}
    regions = None
    _ods_context = _ods_get_default_loc_context(loc)
    _ods_successors = None
    super().__init__(self.build_generic(
      attributes=attributes, results=results, operands=operands,
      successors=_ods_successors, regions=regions, loc=loc, ip=ip))

@_ods_cext.register_operation(_Dialect)
@_ods_extend_opview_class(_ods_ext_module)
class PipelineGetOp(_ods_ir.OpView):
  OPERATION_NAME = "air.pipeline.get"

  _ODS_REGIONS = (0, True)

  def __init__(self, results_, src0, src1, *, loc=None, ip=None):
    operands = []
    results = []
    attributes = {}
    regions = None
    operands.append(_get_op_result_or_value(src0))
    operands.append(_get_op_result_or_value(src1))
    _ods_context = _ods_get_default_loc_context(loc)
    results.extend(results_)
    _ods_successors = None
    super().__init__(self.build_generic(
      attributes=attributes, results=results, operands=operands,
      successors=_ods_successors, regions=regions, loc=loc, ip=ip))

  @builtins.property
  def src0(self):
    return self.operation.operands[0]

  @builtins.property
  def src1(self):
    return self.operation.operands[1]

  @builtins.property
  def results_(self):
    _ods_variadic_group_length = len(self.operation.results) - 1 + 1
    return self.operation.results[0:0 + _ods_variadic_group_length]

@_ods_cext.register_operation(_Dialect)
@_ods_extend_opview_class(_ods_ext_module)
class PipelinePutOp(_ods_ir.OpView):
  OPERATION_NAME = "air.pipeline.put"

  _ODS_REGIONS = (0, True)

  def __init__(self, dst0, dst1, opers, *, loc=None, ip=None):
    operands = []
    results = []
    attributes = {}
    regions = None
    operands.append(_get_op_result_or_value(dst0))
    operands.append(_get_op_result_or_value(dst1))
    operands.extend(_get_op_results_or_values(opers))
    _ods_context = _ods_get_default_loc_context(loc)
    _ods_successors = None
    super().__init__(self.build_generic(
      attributes=attributes, results=results, operands=operands,
      successors=_ods_successors, regions=regions, loc=loc, ip=ip))

  @builtins.property
  def dst0(self):
    return self.operation.operands[0]

  @builtins.property
  def dst1(self):
    return self.operation.operands[1]

  @builtins.property
  def opers(self):
    _ods_variadic_group_length = len(self.operation.operands) - 3 + 1
    return self.operation.operands[2:2 + _ods_variadic_group_length]

@_ods_cext.register_operation(_Dialect)
@_ods_extend_opview_class(_ods_ext_module)
class PipelineStageOp(_ods_ir.OpView):
  OPERATION_NAME = "air.pipeline.stage"

  _ODS_REGIONS = (1, True)

  def __init__(self, results_, opers, *, loc=None, ip=None):
    operands = []
    results = []
    attributes = {}
    regions = None
    operands.extend(_get_op_results_or_values(opers))
    _ods_context = _ods_get_default_loc_context(loc)
    results.extend(results_)
    _ods_successors = None
    super().__init__(self.build_generic(
      attributes=attributes, results=results, operands=operands,
      successors=_ods_successors, regions=regions, loc=loc, ip=ip))

  @builtins.property
  def opers(self):
    _ods_variadic_group_length = len(self.operation.operands) - 1 + 1
    return self.operation.operands[0:0 + _ods_variadic_group_length]

  @builtins.property
  def results_(self):
    _ods_variadic_group_length = len(self.operation.results) - 1 + 1
    return self.operation.results[0:0 + _ods_variadic_group_length]

  @builtins.property
  def body(self):
    return self.regions[0]

@_ods_cext.register_operation(_Dialect)
@_ods_extend_opview_class(_ods_ext_module)
class PipelineTerminatorOp(_ods_ir.OpView):
  OPERATION_NAME = "air.pipeline.terminator"

  _ODS_REGIONS = (0, True)

  def __init__(self, opers, *, loc=None, ip=None):
    operands = []
    results = []
    attributes = {}
    regions = None
    operands.extend(_get_op_results_or_values(opers))
    _ods_context = _ods_get_default_loc_context(loc)
    _ods_successors = None
    super().__init__(self.build_generic(
      attributes=attributes, results=results, operands=operands,
      successors=_ods_successors, regions=regions, loc=loc, ip=ip))

  @builtins.property
  def opers(self):
    _ods_variadic_group_length = len(self.operation.operands) - 1 + 1
    return self.operation.operands[0:0 + _ods_variadic_group_length]

@_ods_cext.register_operation(_Dialect)
@_ods_extend_opview_class(_ods_ext_module)
class PipelineYieldOp(_ods_ir.OpView):
  OPERATION_NAME = "air.pipeline.yield"

  _ODS_REGIONS = (0, True)

  def __init__(self, opers, *, loc=None, ip=None):
    operands = []
    results = []
    attributes = {}
    regions = None
    operands.extend(_get_op_results_or_values(opers))
    _ods_context = _ods_get_default_loc_context(loc)
    _ods_successors = None
    super().__init__(self.build_generic(
      attributes=attributes, results=results, operands=operands,
      successors=_ods_successors, regions=regions, loc=loc, ip=ip))

  @builtins.property
  def opers(self):
    _ods_variadic_group_length = len(self.operation.operands) - 1 + 1
    return self.operation.operands[0:0 + _ods_variadic_group_length]

@_ods_cext.register_operation(_Dialect)
@_ods_extend_opview_class(_ods_ext_module)
class SegmentOp(_ods_ir.OpView):
  OPERATION_NAME = "air.segment"

  _ODS_OPERAND_SEGMENTS = [-1,-1,-1,]

  _ODS_REGIONS = (1, True)

  @builtins.property
  def async_dependencies(self):
    operand_range = _ods_segmented_accessor(
         self.operation.operands,
         self.operation.attributes["operand_segment_sizes"], 0)
    return operand_range

  @builtins.property
  def sizes(self):
    operand_range = _ods_segmented_accessor(
         self.operation.operands,
         self.operation.attributes["operand_segment_sizes"], 1)
    return operand_range

  @builtins.property
  def segment_operands(self):
    operand_range = _ods_segmented_accessor(
         self.operation.operands,
         self.operation.attributes["operand_segment_sizes"], 2)
    return operand_range

  @builtins.property
  def async_token(self):
    return None if len(self.operation.results) < 1 else self.operation.results[0]

  @builtins.property
  def body(self):
    return self.regions[0]

@_ods_cext.register_operation(_Dialect)
@_ods_extend_opview_class(_ods_ext_module)
class SegmentTerminatorOp(_ods_ir.OpView):
  OPERATION_NAME = "air.segment_terminator"

  _ODS_REGIONS = (0, True)

  def __init__(self, *, loc=None, ip=None):
    operands = []
    results = []
    attributes = {}
    regions = None
    _ods_context = _ods_get_default_loc_context(loc)
    _ods_successors = None
    super().__init__(self.build_generic(
      attributes=attributes, results=results, operands=operands,
      successors=_ods_successors, regions=regions, loc=loc, ip=ip))

@_ods_cext.register_operation(_Dialect)
@_ods_extend_opview_class(_ods_ext_module)
class WaitAllOp(_ods_ir.OpView):
  OPERATION_NAME = "air.wait_all"

  _ODS_REGIONS = (0, True)

  def __init__(self, async_token, async_dependencies, *, loc=None, ip=None):
    operands = []
    results = []
    attributes = {}
    regions = None
    operands.extend(_get_op_results_or_values(async_dependencies))
    _ods_context = _ods_get_default_loc_context(loc)
    if async_token is not None: results.append(async_token)
    _ods_successors = None
    super().__init__(self.build_generic(
      attributes=attributes, results=results, operands=operands,
      successors=_ods_successors, regions=regions, loc=loc, ip=ip))

  @builtins.property
  def async_dependencies(self):
    _ods_variadic_group_length = len(self.operation.operands) - 1 + 1
    return self.operation.operands[0:0 + _ods_variadic_group_length]

  @builtins.property
  def async_token(self):
    return None if len(self.operation.results) < 1 else self.operation.results[0]
