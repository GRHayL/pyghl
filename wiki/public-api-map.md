# Public API Map

This map routes Python, CPython, CLI, environment, packaging, and artifact
surfaces to canonical owners while preserving exact evidence states. Source
presence or metadata never proves a built artifact, successful import, execution,
stability, or numerical behavior.

## Evidence state key

- `source-present`: file or symbol exists in inspected source.
- `declared`: a public header, parser, metadata, or Python declaration names it.
- `implemented`: executable source exists.
- `registered`: a CPython table/module initializer or CLI dispatcher wires it.
- `exported`: a Python facade contains an import/attribute route.
- `selected`: build/workflow configuration names it.
- `package-selected`: packaging metadata or copy rules select it.
- `packaged`: a named built wheel/sdist was inspected and contains it.
- `imported`: a named environment loaded it.
- `executed`: a named invocation exercised it.

Only the first seven states are established by baseline source inspection here;
this page records no `packaged`, `imported`, or `executed` result.

## Top-level `pyghl` facade

`src/pyghl/__init__.py` always reaches the following source-visible facade
names. `__all__` explicitly exports `_BINDINGS_AVAILABLE`, `nn`, and
`require_bindings`. `nn` is imported after the extension attempt and its module
defers extension access until binding-dependent calls.

If `from ._pyghl import ...` succeeds, the facade also imports and adds these
exact names to `__all__`:

- Types/error: `Params`, `TabulatedEOS`, `Primitive`, `Conservative`, `Metric`,
  `ADMAux`, `Diagnostics`, `GRHayLError`.
- Functions: `initialize_params`, `initialize_metric`,
  `compute_ADM_auxiliaries`, `initialize_primitives`,
  `initialize_diagnostics`, `compute_conservs`,
  `undensitize_conservatives`, `compute_SU_Bsq_Ssq_BdotS`,
  `limit_v_and_compute_u0`, `limit_utilde_and_compute_v`,
  `guess_primitives`, `tabulated_Palenzuela1D_energy`,
  `tabulated_con2prim_multi_method`, `nn_c2p_guess`, `nn_c2p_guess_x`.
- Constants: `C2P_NONE`, `C2P_NOBLE2D`, `C2P_NOBLE1D`,
  `C2P_NOBLE1D_ENTROPY`, `C2P_NOBLE1D_ENTROPY2`, `C2P_FONT1D`,
  `C2P_PALENZUELA1D`, `C2P_PALENZUELA1D_ENTROPY`, `C2P_NEWMAN1D`,
  `C2P_NEWMAN1D_ENTROPY`.
- Module: `eos`, imported only in the success branch.

These names are `exported` conditionally in source. `_BINDINGS_IMPORT_ERROR` is
private source state and not in `__all__`; `require_bindings()` re-raises a
diagnostic `ImportError` chained from that stored loader failure. Canonical
owners: [extension surface and errors](bindings/extension-surface-and-errors.md)
and [tabulated EOS lifetime and loader](bindings/tabulated-eos-lifetime-and-loader.md).

## CPython extension registration

`csrc/pyghl_module.c::PyInit__pyghl` creates `_pyghl`, readies seven static
types, creates `GRHayLError`, adds those objects and ten integer constants, and
uses `module_methods` as the module function table. This establishes
source-level `registered` wiring, not import evidence.

| Surface family | Exact registered names | Canonical owner |
| --- | --- | --- |
| Seven types | `Params`, `TabulatedEOS`, `Primitive`, `Conservative`, `Metric`, `ADMAux`, `Diagnostics` | [Extension surface and errors](bindings/extension-surface-and-errors.md) |
| Exception | `GRHayLError` | [Extension surface and errors](bindings/extension-surface-and-errors.md) |
| Constructors/factories | `initialize_params`, `initialize_tabulated_eos_functions_and_params`, `initialize_metric`, `initialize_primitives`, `initialize_diagnostics` | [Bindings hub](bindings/index.md); tabulated initializer belongs to [EOS lifetime](bindings/tabulated-eos-lifetime-and-loader.md) |
| Metric/conservative helpers | `compute_ADM_auxiliaries`, `compute_conservs`, `undensitize_conservatives`, `compute_SU_Bsq_Ssq_BdotS` | [Extension surface and errors](bindings/extension-surface-and-errors.md) |
| Velocity/guess helpers | `limit_v_and_compute_u0`, `limit_utilde_and_compute_v`, `guess_primitives` | [Extension surface and errors](bindings/extension-surface-and-errors.md) |
| C2P calls | `tabulated_Palenzuela1D_energy`, `tabulated_con2prim_multi_method` | [Extension surface and errors](bindings/extension-surface-and-errors.md) |
| NN calls | `nn_c2p_guess`, `nn_c2p_guess_x` | [Extension surface and errors](bindings/extension-surface-and-errors.md) |
| Integer constants | same ten `C2P_*` names listed in top-level facade | [Extension surface and errors](bindings/extension-surface-and-errors.md) |
| EOS instance methods | `tabulated_enforce_bounds_rho_Ye_T`, `tabulated_enforce_bounds_rho_Ye_eps`, `tabulated_compute_P_from_T`, `tabulated_compute_eps_from_T`, `tabulated_compute_cs2_from_T`, `tabulated_compute_P_eps_from_T`, `tabulated_compute_P_eps_S_from_T`, `tabulated_compute_T_from_eps`, `tabulated_compute_P_T_from_eps`, `load_nn_c2p_hdf5`, `close` | [EOS lifetime](bindings/tabulated-eos-lifetime-and-loader.md) |

The low-level tabulated initializer is registered on `_pyghl` but is not
top-level-imported by `pyghl.__init__`; `pyghl.eos` wraps it after resolving a
table path. Upstream declarations without a C registration and facade route are
`upstream-only`; see [GRHayL integration](integration/grhayl-submodule.md).

## Pure-Python NN facade

`src/pyghl/nn.py` is always exposed as `pyghl.nn`. It has no `__all__`; its
source-defined facade families route as follows:

| Source-visible family | Names | Canonical owner |
| --- | --- | --- |
| Data records/readers | `DatasetPoint`, `NNGuessInput`, `NNGuess`, `read_dataset_header`, `iter_dataset_points`, `read_training_dataset` | [Dataset and generation](nn/dataset-and-generation.md) |
| Binding-dependent guessing | `guess`, `guess_x`, `flat_metric`, `nn_initial_guess` | [Inference and export](nn/inference-and-export.md) |
| Training/export facade | `train_regressor`, `train_on_dataset`, `export_to_c_header`, `export_to_hdf5` | [Training and checkpoints](nn/training-and-checkpoints.md) |
| EOS/model HDF5 facade | `append_to_eos_file`, `append_matching_installed_to_eos_file`, `remove_from_eos_file`, `eos_nn_metadata`, `installed_nn_models`, `find_matching_installed_model`, `install_nn_model` | [Model/EOS HDF5 lifecycle](nn/model-eos-hdf5-lifecycle.md) |
| Inference bundles | `save_inference_bundle`, `load_inference_bundle` | [Inference and export](nn/inference-and-export.md) |

Function definitions are `source-present` and `implemented`; a call can still
fail at deferred extension or dependency loading. `src/pyghl/nn_c2p/__init__.py`
declares no additional facade exports.

## CLI and package entry point

`pyproject.toml` has `[project.scripts] pyghl = "pyghl.cli:main"`; this is
`package-selected`, not proof that a named artifact contains or installs the
script.

| User spelling | Parser declaration | Dispatch implementation | Owner |
| --- | --- | --- | --- |
| no arguments, `-h`, `--help` | root parser/source help | manual early branch; no arguments returns `2`, explicit help returns `0` | [CLI hub](cli/index.md) |
| `-v`, `--version` | root option declared | manual early branch | [CLI hub](cli/index.md) |
| `train` | subparser declared | requires bindings, then delegates to `nn_c2p_train.main` | [Command workflows](cli/command-workflows.md) |
| `append` | subparser declared | delegates to `append_eos_file.main` | [Model/EOS HDF5 lifecycle](nn/model-eos-hdf5-lifecycle.md) |
| `append-eos` | not declared by `build_parser` | accepted as dispatcher alias | [Command workflows](cli/command-workflows.md) |
| `check-eos` | subparser declared | delegates to `check_eos.main` | [Model/EOS HDF5 lifecycle](nn/model-eos-hdf5-lifecycle.md) |
| `list-models` | subparser declared | delegates to `list_installed_models.main` | [Model/EOS HDF5 lifecycle](nn/model-eos-hdf5-lifecycle.md) |
| `remove-eos-nn` | subparser declared | delegates to `remove_eos_nn.main` | [Model/EOS HDF5 lifecycle](nn/model-eos-hdf5-lifecycle.md) |
| other spelling | not declared | `parser.error` | [CLI hub](cli/index.md) |

First-level registration/dispatch says nothing about delegated parser success,
filesystem mutation, network access, training, or exit status in a runtime.

## Environment surface

| Name | Read/used by | Exact effect | State and owner |
| --- | --- | --- | --- |
| `GRHAYL_DIR` | `setup.py::_default_grhayl_root`; `src/pyghl/__init__.py::require_bindings` | selects external GRHayL build root; also selects diagnostic library directory when a loader message names `libghl` | `implemented`; [GRHayL integration](integration/grhayl-submodule.md) |
| `GRHAYL_CONFIGURE_ARGS` | `setup.py::_run_make_grhayl` | shell-split extra arguments appended after `--prefix=.` when configure runs | `implemented`; [Build/package/release/CI](build-package-release-ci.md) |
| `GRHAYL_EOS_TABLE_DIR` | `src/pyghl/eos.py::_resolve_table_path` | basename fallback directory for exact name, `.h5`, then `.hdf5` | `implemented`; [EOS lifetime](bindings/tabulated-eos-lifetime-and-loader.md) |
| `LD_LIBRARY_PATH` | emitted by `require_bindings` diagnostic | suggested local/editable remediation when original import error contains `libghl`; package code does not assign it | diagnostic text only; [EOS lifetime](bindings/tabulated-eos-lifetime-and-loader.md) |

## Artifact and packaging surface

| Artifact family | Source/configuration evidence | Strongest baseline state | Canonical owner |
| --- | --- | --- | --- |
| `pyghl._pyghl` extension | `setup.py::ext_modules` | `package-selected`; not compiled/imported here | [Build/package/release/CI](build-package-release-ci.md) |
| `libghl*.so*`, `libghl*.dylib` | `BuildExt.run` copy list/globs and `pyproject.toml` package-data patterns | `package-selected`; named artifact inspection required | [Build/package/release/CI](build-package-release-ci.md) |
| `src/pyghl/py.typed` | tracked marker under package tree | `source-present`; named package inspection required | [Build/package/release/CI](build-package-release-ci.md) |
| `nn_c2p/models/*.h5` | four tracked files plus package-data glob | `source-present` and `package-selected`; no schema/validity claim here | [Model/EOS HDF5 lifecycle](nn/model-eos-hdf5-lifecycle.md) |
| EOS HDF5 and embedded `grhayl_nn_c2p` | producer/consumer functions in `_nn_hdf5.py` and GRHayL | `implemented`; user-owned/mutable actual artifact | [Model/EOS HDF5 lifecycle](nn/model-eos-hdf5-lifecycle.md) |
| binary training dataset | readers and generator source | `implemented`; generated/user-owned | [Dataset and generation](nn/dataset-and-generation.md) |
| checkpoint, inference bundle, generated C header, standalone NN HDF5 | `_nn_train.py`, `_nn_infer.py`, `_nn_hdf5.py`, conversion command | `implemented`; generated/user-owned until an artifact is inspected | [Generated boundaries](generated-boundaries.md) |
| sdist/wheel | `pyproject.toml`, `MANIFEST.in`, `setup.py`, workflow selection | `selected`/`package-selected`; no named artifact inspected here | [Build/package/release/CI](build-package-release-ci.md) |

## Change impact

- `src/pyghl/__init__.py`: review conditional export rows and loader owner.
- `csrc/pyghl_module.c` registration/type/method tables: review CPython rows,
  bindings leaves, parent-pin traces, and missing direct wrapper proof.
- `src/pyghl/cli.py` or `[project.scripts]`: review command and package-entry
  states independently.
- environment reads or artifact producers: add one owner route; do not promote
  metadata or source presence into runtime/artifact claims.
