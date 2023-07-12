from triton_air import _air_ops_gen as air_dialect
from triton_air.util import mlir_mod_ctx

with mlir_mod_ctx(allow_unregistered_dialects=True) as module:
    x = air_dialect.ChannelOp("bob")

print(module)
