# pyghl

This directory contains Python bindings for GRHayL, including enough tabulated-EOS
and con2prim functionality to reproduce the neural-network dataset/testing
generators in `Unit_Tests/data_gen`.

## What is included

- `pyghl.initialize_params(...)`
- `pyghl.initialize_metric(...)`
- `pyghl.compute_ADM_auxiliaries(...)`
- `pyghl.initialize_primitives(...)`
- `pyghl.initialize_diagnostics(...)`
- `pyghl.compute_conservs(...)`
- `pyghl.undensitize_conservatives(...)`
- `pyghl.compute_SU_Bsq_Ssq_BdotS(...)`
- `pyghl.limit_v_and_compute_u0(...)`
- `pyghl.limit_utilde_and_compute_v(...)`
- `pyghl.guess_primitives(...)`
- `pyghl.tabulated_Palenzuela1D_energy(...)`
- `pyghl.nn_c2p_guess(eos, ...)`
- `pyghl.nn_c2p_guess_x(eos, ...)`
- Struct wrappers:
  - `pyghl.Primitive`
  - `pyghl.Conservative`
  - `pyghl.Metric`
  - `pyghl.ADMAux`
  - `pyghl.Diagnostics`
- `pyghl.eos.initialize_tabulated_eos_functions_and_params(...)`
- `pyghl.nn` helpers:
  - `flat_metric()`
  - `guess(eos, ...)`
  - `guess_x(eos, ...)`
  - `nn_initial_guess(...)`
  - `iter_dataset_points(...)`
  - `read_training_dataset(...)`
  - `train_regressor(...)`
  - `train_on_dataset(...)`
  - `save_inference_bundle(...)`
  - `load_inference_bundle(...)`
  - `export_to_c_header(...)`
- `TabulatedEOS` methods:
  - `tabulated_enforce_bounds_rho_Ye_T`
  - `tabulated_enforce_bounds_rho_Ye_eps`
  - `tabulated_compute_P_from_T`
  - `tabulated_compute_eps_from_T`
  - `tabulated_compute_cs2_from_T`
  - `tabulated_compute_P_eps_from_T`
  - `tabulated_compute_P_eps_S_from_T`
  - `tabulated_compute_T_from_eps`
  - `tabulated_compute_P_T_from_eps`

## Build requirements

- The pinned GRHayL submodule initialized at `extern/GRHayL`, or `GRHAYL_DIR`
  set to a configured GRHayL checkout.
- HDF5 enabled in GRHayL build.
- Python build tooling (`pip`, `setuptools`, compiler toolchain).

## Install (from repo root)

Initialize the pinned GRHayL source:

```bash
git submodule update --init --recursive
```

Standard:

```bash
python3 -m pip install -e .
```

With neural-network training helpers:

```bash
python3 -m pip install -e '.[nn]'
```

Offline or restricted-network environments:

```bash
python3 -m pip install --no-build-isolation -e .
```

The build step runs:

```bash
make -C extern/GRHayL grhayl
```

and then compiles the Python extension against `extern/GRHayL/build/lib/libghl.so`.
Set `GRHAYL_DIR=/path/to/GRHayL` to build against a different local checkout.

If editable install is not available in your environment, you can work from the
repository root with:

```bash
PYTHONPATH=src python3 ...
```

## Example

```python
import pyghl as ghl

params = ghl.initialize_params()
eos = ghl.eos.initialize_tabulated_eos_functions_and_params(
    params,
    "Unit_Tests/sample_table/Hempel_SFHoEOS_rho222_temp180_ye60_version_1.1_20120817_simple.h5",
)

rho = 1e-12
Y_e = 0.05
T = 1e2

rho, Y_e, T = eos.tabulated_enforce_bounds_rho_Ye_T(rho, Y_e, T)
P = eos.tabulated_compute_P_from_T(rho, Y_e, T)
```

You can also pass a short table name and set `GRHAYL_EOS_TABLE_DIR`.

## Neural-network data generators

The following example scripts mirror the C generators in
`Unit_Tests/data_gen`:

- `examples/nn_c2p_generate_dataset.py`
- `examples/nn_c2p_test.py`
- `examples/nn_c2p_train.py`

The same tools are also importable/runnable as package modules:

- `python -m pyghl.nn_c2p.nn_c2p_generate_dataset ...`
- `python -m pyghl.nn_c2p.nn_c2p_train ...`
- `python -m pyghl.nn_c2p.nn_c2p_test ...`
- `python -m pyghl.nn_c2p.append_eos_file ...`
- `python -m pyghl.nn_c2p.check_eos ...`
- `python -m pyghl.nn_c2p.list_installed_models`
- `python -m pyghl.nn_c2p.remove_eos_nn ...`

Example:

```bash
PYTHONPATH=src python3 examples/nn_c2p_generate_dataset.py \
  Unit_Tests/sample_table/Hempel_SFHoEOS_rho222_temp180_ye60_version_1.1_20120817_simple.h5 \
  train --n-pts 2 --output /tmp/nn_training_dataset.bin

PYTHONPATH=src python3 examples/nn_c2p_test.py \
  Unit_Tests/sample_table/Hempel_SFHoEOS_rho222_temp180_ye60_version_1.1_20120817_simple.h5 \
  /tmp/nn_training_dataset.bin --limit 8

python3 -m pyghl.nn_c2p.nn_c2p_generate_dataset \
  Unit_Tests/sample_table/Hempel_SFHoEOS_rho222_temp180_ye60_version_1.1_20120817_simple.h5 \
  train

python3 -m pyghl.nn_c2p.nn_c2p_train \
  Unit_Tests/sample_table/Hempel_SFHoEOS_rho222_temp180_ye60_version_1.1_20120817_simple.h5

python3 -m pyghl.nn_c2p.nn_c2p_train \
  Unit_Tests/sample_table/Hempel_SFHoEOS_rho222_temp180_ye60_version_1.1_20120817_simple.h5 \
  /tmp/nn_training_dataset.bin

python3 -m pyghl.nn_c2p.append_eos_file \
  Unit_Tests/sample_table/Hempel_SFHoEOS_rho222_temp180_ye60_version_1.1_20120817_simple.h5 \
  /tmp/tiny_mlp_model.h5

python3 -m pyghl.nn_c2p.append_eos_file \
  Unit_Tests/sample_table/Hempel_SFHoEOS_rho222_temp180_ye60_version_1.1_20120817_simple.h5

python3 -m pyghl.nn_c2p.list_installed_models

python3 -m pyghl.nn_c2p.check_eos \
  Unit_Tests/sample_table/Hempel_SFHoEOS_rho222_temp180_ye60_version_1.1_20120817_simple.h5

python3 -m pyghl.nn_c2p.remove_eos_nn \
  Unit_Tests/sample_table/Hempel_SFHoEOS_rho222_temp180_ye60_version_1.1_20120817_simple.h5
```
