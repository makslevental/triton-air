from triton_mlir_bindings.dialects import arith, triton as tt
from triton_mlir_bindings.ir import (
    IntegerType,
)
from triton_mlir_bindings.util.utils import (
    mlir_mod_ctx,
)

from triton_air import _air_ops_gen as air_dialect
from util import filecheck


class Test:
    def test_smoke(self):
        with mlir_mod_ctx(allow_unregistered_dialects=True) as (
            context,
            location,
            module,
            ip,
        ):
            air_dialect.ChannelOp("bob")
        correct = """\
        module {
          "air.channel"() {sym_name = "bob"} : () -> ()
        }
        """
        filecheck(correct, module)

    def test_smoke_with_triton(self):
        with mlir_mod_ctx(allow_unregistered_dialects=True) as (
            context,
            location,
            module,
            ip,
        ):
            tt.register_dialect(context)
            i32 = IntegerType.get_signless(32)

            @tt.FuncOp.from_py_func(results=[])
            def kernel_0123():
                c64 = arith.ConstantOp(i32, 64)
                v0 = tt.GetProgramIdOp(axis=0)
                air_dialect.ChannelOp("bob")
                tt.ReturnOp([])

        correct = """\
        module {
          tt.func @kernel_0123() {
            %c64_i32 = arith.constant 64 : i32
            %0 = tt.get_program_id {axis = 0 : i32} : i32
            "air.channel"() {sym_name = "bob"} : () -> ()
            tt.return
          }
        }
        """
        filecheck(correct, module)
