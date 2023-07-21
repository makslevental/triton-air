from mlir_utils import DefaultContext
from triton_mlir_bindings.dialects import triton as triton_dialect

triton_dialect.register_dialect(DefaultContext)

from .dialects.ext.triton import TritonTensor
