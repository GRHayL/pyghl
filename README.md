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
- `pyghl append-eos` embeds a trained model into an EOS HDF5 file.

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

## First Successful Run

The quickest useful workflow is to train a small model for an EOS table and
append the result back into that EOS file.

### 1. Train a model

```bash
pyghl train path/to/eos_table.h5
```

This creates training data if no dataset is supplied, trains a small neural
network, writes model artifacts such as `tiny_mlp_model.h5`, and registers the
installed model by the EOS canonical MD5 hash.

To train from an existing dataset:

```bash
pyghl train path/to/eos_table.h5 path/to/nn_training_dataset.bin
```

### 2. Append a model to an EOS file

Append an explicit model:

```bash
pyghl append-eos path/to/eos_table.h5 tiny_mlp_model.h5
```

Or append the installed model matching the EOS canonical MD5 hash:

```bash
pyghl append-eos path/to/eos_table.h5
```

### 3. Inspect installed models

```bash
python -m pyghl.nn_c2p.list_installed_models
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
pyghl append-eos <eos-file> [nn-hdf5]
pyghl --version
```

Additional module entry points:

```bash
python -m pyghl.nn_c2p.nn_c2p_generate_dataset ...
python -m pyghl.nn_c2p.nn_c2p_train ...
python -m pyghl.nn_c2p.nn_c2p_test ...
python -m pyghl.nn_c2p.append_eos_file ...
python -m pyghl.nn_c2p.check_eos ...
python -m pyghl.nn_c2p.list_installed_models
python -m pyghl.nn_c2p.remove_eos_nn ...
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

If editable installs are not available in your environment, many Python-only
checks can be run from the repository root with:

```bash
PYTHONPATH=src python -m pyghl.nn_c2p.list_installed_models
```
