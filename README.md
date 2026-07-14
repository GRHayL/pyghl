# pyghl

Python bindings and neural-network tooling for
[GRHayL](https://github.com/GRHayL/GRHayL).

`pyghl` packages a pinned GRHayL build together with a CPython extension and
Python utilities for tabulated equations of state, conservative-to-primitive
experiments, and neural-network initial guesses.

> **Warning**
>
> While revised by humans, the majority of the code in this repository has
> leveraged AI assistance. Users should review results carefully and use this
> package at their own risk.

At a high level:

- `pyghl` exposes selected GRHayL C structs and routines to Python.
- `pyghl.eos` provides tabulated-EOS loading and interpolation helpers.
- `pyghl.nn` provides neural-network con2prim data, training, export, and EOS
  embedding helpers.
- `pyghl train` trains a neural-network model for an EOS table.
- `pyghl append` embeds a trained model into an EOS HDF5 file.

## Installation

For end users installing from PyPI:

```bash
python -m pip install pyghl
```

Check the installed package:

```bash
pyghl --version
```

For contributors working from a clone of this repository:

```bash
git clone https://github.com/GRHayL/pyghl.git
cd pyghl
git submodule update --init --recursive
python -m pip install -e .
```

The local build compiles the pinned GRHayL submodule in `extern/GRHayL` and then
builds the Python extension against the resulting `libghl`.

If you need to build against a different local GRHayL checkout:

```bash
GRHAYL_DIR=/path/to/GRHayL python -m pip install -e .
```

## Prerequisites by Workflow

`python -m pip install pyghl` should install a prebuilt wheel on supported
platforms. Source builds need additional system tools:

- Local extension builds: a C compiler, `make`, HDF5, and Python build tooling.
- Editable contributor installs: initialized `extern/GRHayL` submodule.
- Neural-network training: `torch`, `numpy`, and `h5py` from package
  dependencies.
- EOS workflows: a GRHayL-compatible tabulated EOS HDF5 file.

Current CI builds wheels for Linux x86_64 and macOS arm64.

> **EOS format support**
>
> Currently, `pyghl` supports only tables in the StellarCollapse HDF5 format.
> Support for tables in the CompOSE format is planned for a future release.

## Downloading an EOS Table

[stellarcollapse.org](https://stellarcollapse.org/equationofstate) provides
GRHayL-compatible tabulated EOS files. For example, the
[APR EOS page](https://stellarcollapse.org/APREOS.html) provides an APR table
with NSE (3335 nuclides). Download and extract it before passing the HDF5 file
to `pyghl`:

```bash
curl -fL \
  'https://stockholmuniversity.box.com/s/xatxe62v9ywxr5sl2vf94e180uf1zkik?download=1' \
  -o APR_3335_rho393_temp133_ye66_gitM180edd5_20190225.h5.tar.bz2
tar -xjf APR_3335_rho393_temp133_ye66_gitM180edd5_20190225.h5.tar.bz2
```

This APR table does not currently have an installed `pyghl` model, so train
one and embed it in the extracted EOS file:

```bash
pyghl train APR_3335_rho393_temp133_ye66_gitM180edd5_20190225.h5
pyghl check-eos APR_3335_rho393_temp133_ye66_gitM180edd5_20190225.h5
```

`pyghl train` generates a training dataset, trains and saves the model, and
embeds the resulting neural-network data into the EOS file.

## Basic EOS Neural-Network Workflow

The quickest useful workflow is to make sure the EOS file itself contains the
neural-network data that GRHayL loads when `enable_neural_net_c2p=True`.

### 1. Try the installed model cache first

```bash
pyghl append path/to/eos_table.h5
```

This looks for an installed neural-network model whose recorded EOS hash matches
the EOS file, then embeds it into the EOS under the `grhayl_nn_c2p` group. This
is the preferred first command because it is fast and avoids retraining known
EOS tables.

### 2. Train if the EOS is unknown

```bash
pyghl train path/to/eos_table.h5
```

If no installed model matches, `pyghl train` creates training data when no
dataset is supplied, trains a small neural network, writes model artifacts such
as `tiny_mlp_model.h5`, registers the model by the EOS canonical MD5 hash, and
embeds the result into the EOS file by default.

To train from an existing dataset:

```bash
pyghl train path/to/eos_table.h5 path/to/nn_training_dataset.bin
```

### 3. Retrain a known EOS deliberately

```bash
pyghl train path/to/eos_table.h5 --force_retrain
```

Use this when the EOS is already known but you want to replace the installed
model artifacts with a fresh training run. If the EOS already contains embedded
neural-network data and you want to replace it too, add `--overwrite_eos`.

### Other Useful Commands

Inspect whether an EOS file already contains embedded neural-network data:

```bash
pyghl check-eos path/to/eos_table.h5
```

List installed models that can be matched by EOS hash:

```bash
pyghl list-models
```

Append a specific model file instead of using the installed model cache:

```bash
pyghl append path/to/eos_table.h5 path/to/tiny_mlp_model.h5
```

Remove embedded neural-network data from an EOS file:

```bash
pyghl remove-eos-nn path/to/eos_table.h5
```

## Python API Example

```python
import pyghl as ghl

params = ghl.initialize_params()
eos = ghl.eos.initialize_tabulated_eos_functions_and_params(
    params,
    "path/to/eos_table.h5",
)

rho = 1.0e-12
Y_e = 0.05
T = 1.0e2

rho, Y_e, T = eos.tabulated_enforce_bounds_rho_Ye_T(rho, Y_e, T)
pressure = eos.tabulated_compute_P_from_T(rho, Y_e, T)
print(pressure)
```

## Command-Line Tools

Primary commands:

```bash
pyghl train <eos-file> [dataset]
pyghl append <eos-file> [nn-hdf5]
pyghl check-eos <eos-file>
pyghl list-models
pyghl remove-eos-nn <eos-file>
pyghl --version
```

## What Is Exposed?

Selected top-level bindings include:

- `initialize_params`
- `initialize_metric`
- `compute_ADM_auxiliaries`
- `initialize_primitives`
- `initialize_diagnostics`
- `compute_conservs`
- `undensitize_conservatives`
- `compute_SU_Bsq_Ssq_BdotS`
- `limit_v_and_compute_u0`
- `limit_utilde_and_compute_v`
- `guess_primitives`
- `tabulated_Palenzuela1D_energy`
- `tabulated_con2prim_multi_method`
- `nn_c2p_guess`
- `nn_c2p_guess_x`

Struct wrappers include:

- `Primitive`
- `Conservative`
- `Metric`
- `ADMAux`
- `Diagnostics`
- `Params`
- `TabulatedEOS`

## Repository Map

- `csrc/`: CPython extension source.
- `src/pyghl/`: Python package.
- `src/pyghl/nn_c2p/`: command-line neural-network workflows.
- `examples/`: direct example scripts.
- `extern/GRHayL/`: pinned GRHayL submodule used for builds.
- `.github/workflows/`: wheel and PyPI publishing automation.
- `PUBLISHING.md`: release and PyPI publishing checklist.

## Contributor Setup

Recommended local workflow:

```bash
git submodule update --init --recursive
python -m pip install -e .
python -m compileall -q src setup.py
```

Build a local wheel:

```bash
python -m pip wheel . -w /tmp/pyghl-wheel --no-deps --no-build-isolation
```
