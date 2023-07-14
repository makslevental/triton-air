from triton_mlir_bindings.dialects import triton as triton_dialect
from triton_mlir_bindings.ir import Value


def maybe_cast(val: Value):
    # doesn't work until triton catches up to
    # https://github.com/llvm/llvm-project/commit/bfb1ba752655bf09b35c486f6cc9817dbedfb1bb
    # if Scalar.isinstance(val):
    #     return Scalar(val)
    return val


from mlir_utils.dialects import util

util.maybe_cast = maybe_cast

from mlir_utils import DefaultContext

triton_dialect.register_dialect(DefaultContext)
