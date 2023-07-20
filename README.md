# triton_air

## Install

```shell
pip install triton-air \
  -f https://github.com/makslevental/triton-air/releases/expanded_assets/latest \
  -f https://github.com/makslevental/triton/releases/expanded_assets/2.1.0 \
  -f https://github.com/makslevental/triton/releases/expanded_assets/2.1.0 \
  -f https://github.com/makslevental/mlir-python-utils/releases/expanded_assets/latest \
  -f https://github.com/makslevental/mlir-wheels/releases/expanded_assets/latest
```

## Dev

```shell
# you need setuptools >= 64 for build_editable
$ pip3 install setuptools -U
$ pip install -e . \
  -f https://github.com/makslevental/triton/releases/expanded_assets/2.1.0 \
  -f https://github.com/makslevental/mlir-python-utils/releases/expanded_assets/latest \
  -f https://github.com/makslevental/mlir-wheels/releases/expanded_assets/latest
$ configure-mlir-python-utils triton_mlir_bindings
```