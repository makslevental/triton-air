# triton_air

## Install

```shell
export PIP_EXTRA_INDEX_URL=https://github.com/makslevental/wheels/releases/expanded_assets/i
$ pip install triton-air
$ configure-mlir-python-utils triton_mlir_bindings
```

## Dev

```shell
export PIP_EXTRA_INDEX_URL=https://github.com/makslevental/wheels/releases/expanded_assets/i
# you need setuptools >= 64 for build_editable
$ pip3 install setuptools -U
$ pip install -e . 
$ configure-mlir-python-utils triton_mlir_bindings
```