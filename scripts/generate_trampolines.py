import ast
import copy
import inspect
import re
from typing import Any

import black
from triton_mlir_bindings.dialects import _tt_ops_gen, _arith_ops_gen

from triton_air.dialects import (
    _air_ops_gen,
    _triton,
    _air,
    _arith,
)


def ast_call(name, args=None, keywords=None):
    if keywords is None:
        keywords = []
    if args is None:
        args = []
    return ast.Call(
        func=ast.Name(id=name, ctx=ast.Load()),
        args=args,
        keywords=keywords,
    )


RESERVED_KEYWORDS = {"return": "return_", "assert": "assert_"}


class CollectAllOps(ast.NodeVisitor):
    camel_to_snake_pat = re.compile(r"(?<!^)(?=[A-Z])")

    def __init__(self, skips=None):
        self.skips = {"_Dialect"}
        if skips is not None:
            self.skips.update(skips)
        self.ops = []
        self.imports = []

    def visit_ClassDef(self, node: ast.ClassDef) -> Any:
        if node.name in self.skips:
            return

        self.imports.append(ast.alias(node.name))

        init_fn = next(n for n in node.body if isinstance(n, ast.FunctionDef))
        args = init_fn.args
        args.args.pop(0)
        for a in args.args:
            a.arg = self.camel_to_snake_pat.sub("_", a.arg).lower()

        for k in args.kwonlyargs:
            k.arg = self.camel_to_snake_pat.sub("_", k.arg).lower()

        keywords = [
            ast.keyword(k.arg, ast.Name(k.arg))
            for k, d in zip(args.kwonlyargs, args.kw_defaults)
        ]

        fun_name = self.camel_to_snake_pat.sub("_", node.name.replace("Op", "")).lower()
        fun_name = RESERVED_KEYWORDS.get(fun_name, fun_name)
        n = ast.FunctionDef(
            name=fun_name,
            args=copy.deepcopy(args),
            body=[ast.Return(ast_call(node.name, args.args, keywords))],
            decorator_list=[],
        )
        ast.fix_missing_locations(n)
        self.ops.append(n)


def main(intput_file_path, output_file_path, import_from_module, skips=None):
    input_file = open(intput_file_path, "r").read()
    tree = ast.parse(input_file)
    collect_all = CollectAllOps(skips=skips)
    collect_all.visit(tree)
    imports = ast.ImportFrom(
        module=import_from_module,
        names=collect_all.imports,
        level=0,
    )
    new_mod = ast.Module([imports] + collect_all.ops, [])
    new_src = ast.unparse(new_mod)
    formatted_new_src = black.format_file_contents(
        new_src, fast=False, mode=black.Mode()
    )
    output_file = open(output_file_path, "w")
    output_file.write(formatted_new_src)


if __name__ == "__main__":
    main(
        inspect.getfile(_tt_ops_gen),
        inspect.getfile(_triton),
        "triton_mlir_bindings.dialects.triton",
    )

    main(
        inspect.getfile(_air_ops_gen),
        inspect.getfile(_air),
        "._air_ops_gen",
    )

    main(
        inspect.getfile(_arith_ops_gen),
        inspect.getfile(_arith),
        "triton_mlir_bindings.dialects.arith",
        skips={"ConstantOp"},
    )
