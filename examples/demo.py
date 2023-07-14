from triton_mlir_bindings.util.utils import mlir_mod_ctx

from triton_air.dialects import air

with mlir_mod_ctx(allow_unregistered_dialects=True) as ctx:
    x = air.channel("bob")

print(ctx)
