from triton_mlir_bindings.dialects import triton as triton_dialect


from mlir_utils import DefaultContext

triton_dialect.register_dialect(DefaultContext)
