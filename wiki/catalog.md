# Catalog and Alias Router

This page owns term and alias routing. Each row selects one canonical owner;
definitions and behavior stay in that owner and exact repository authority.

## Terms and Aliases

| Term or alias | Canonical owner | Primary authority | Proof/workflow route |
| --- | --- | --- | --- |
| `ADMAux`, `Conservative`, `Diagnostics`, `Metric`, `Params`, `Primitive` | [Extension surface and errors](bindings/extension-surface-and-errors.md) | `csrc/pyghl_module.c` type/get-set tables | [Public API map](public-api-map.md), [test map](test-map.md) |
| `TabulatedEOS`, EOS close/deallocation, interpolation methods | [Tabulated EOS lifetime and loader](bindings/tabulated-eos-lifetime-and-loader.md) | `csrc/pyghl_module.c` EOS wrapper | [Bindings hub](bindings/index.md), [test map](test-map.md) |
| `GRHayLError`, wrapper validation, error translation | [Extension surface and errors](bindings/extension-surface-and-errors.md) | `csrc/pyghl_module.c` | [Public API map](public-api-map.md) |
| `_pyghl`, binding import, `require_bindings` | [Tabulated EOS lifetime and loader](bindings/tabulated-eos-lifetime-and-loader.md) | `src/pyghl/__init__.py`, dynamic loader | [Build/package/release/CI](build-package-release-ci.md) |
| `GRHAYL_DIR`, `GRHAYL_CONFIGURE_ARGS`, `BUILD_LIB_DIR`, `libghl` | [Build/package/release/CI](build-package-release-ci.md) | `setup.py` | [GRHayL integration](integration/grhayl-submodule.md), [generated boundaries](generated-boundaries.md) |
| `GRHAYL_EOS_TABLE_DIR`, EOS basename/path resolution | [Tabulated EOS lifetime and loader](bindings/tabulated-eos-lifetime-and-loader.md) | `src/pyghl/eos.py` | [Workflows](workflows.md) |
| GRHayL pin, gitlink, submodule, advanced checkout, external GRHayL checkout | [GRHayL integration](integration/grhayl-submodule.md) | Parent gitlink, `.gitmodules`, `setup.py` | [Change impact](change-impact.md) |
| C2P constants, wrapper functions, initialization helpers | [Extension surface and errors](bindings/extension-surface-and-errors.md) | `csrc/pyghl_module.c` registration tables | [Public API map](public-api-map.md) |
| dataset header, `<QQQ`, `<16f`, dataset point, `q/r/s/t/x` | [Dataset and generation](nn/dataset-and-generation.md) | `src/pyghl/_nn_dataset.py`, `src/pyghl/nn_c2p/nn_c2p_generate_dataset.py` | [Generated boundaries](generated-boundaries.md) |
| dataset generation, `x_correction`, `x_best_correction`, scan points | [Dataset and generation](nn/dataset-and-generation.md) | `src/pyghl/nn_c2p/nn_c2p_generate_dataset.py` | [CLI workflows](cli/command-workflows.md) |
| feature transform, robust min-max, `TinyMLP_Logit`, training split | [Training and checkpoints](nn/training-and-checkpoints.md) | `src/pyghl/_nn_common.py`, `src/pyghl/_nn_train.py` | [Test map](test-map.md) |
| checkpoint, resume, training log, early stopping, `perm_sha1_16` | [Training and checkpoints](nn/training-and-checkpoints.md) | `src/pyghl/_nn_train.py` | [Generated boundaries](generated-boundaries.md) |
| inference bundle, `weights_only`, target decode, `cons_to_x_guess` | [Inference and export](nn/inference-and-export.md) | `src/pyghl/_nn_infer.py` | [Generated boundaries](generated-boundaries.md) |
| C header, header guard SHA-1, `header_to_hdf5` | [Inference and export](nn/inference-and-export.md) | `src/pyghl/_nn_train.py`, `src/pyghl/nn_c2p/header_to_hdf5.py` | [Generated boundaries](generated-boundaries.md) |
| standalone NN HDF5, model v2, `dims/meta/scaling/layers/audit` | [Model/EOS HDF5 lifecycle](nn/model-eos-hdf5-lifecycle.md) | `src/pyghl/_nn_hdf5.py` | [Generated boundaries](generated-boundaries.md) |
| `grhayl_nn_c2p`, embedded NN, append/remove/overwrite | [Model/EOS HDF5 lifecycle](nn/model-eos-hdf5-lifecycle.md) | `src/pyghl/_nn_hdf5.py` | [CLI workflows](cli/command-workflows.md) |
| canonical EOS MD5, raw file MD5, model match, installed model cache | [Model/EOS HDF5 lifecycle](nn/model-eos-hdf5-lifecycle.md) | `src/pyghl/_nn_hdf5.py` | [Generated boundaries](generated-boundaries.md) |
| `pyghl train`, force retrain, overwrite installed model | [CLI command workflows](cli/command-workflows.md) | `src/pyghl/cli.py`, `src/pyghl/nn_c2p/nn_c2p_train.py` | [Training owner](nn/training-and-checkpoints.md) |
| `pyghl append`, hidden `append-eos` dispatch alias | [CLI command workflows](cli/command-workflows.md) | `src/pyghl/cli.py`, `src/pyghl/nn_c2p/append_eos_file.py` | [HDF5 lifecycle](nn/model-eos-hdf5-lifecycle.md) |
| `pyghl check-eos`, `list-models`, `remove-eos-nn` | [CLI command workflows](cli/command-workflows.md) | `src/pyghl/cli.py`, delegated modules | [Public API map](public-api-map.md) |
| `pyghl -v`, `pyghl --version` | [CLI hub](cli/index.md) | `src/pyghl/cli.py::_package_version`, `src/pyghl/cli.py::main` | [Public API map](public-api-map.md) |
| StellarCollapse catalog, category/table picker, curses UI | [EOS catalog and download](cli/eos-catalog-and-download.md) | `src/pyghl/nn_c2p/eos_catalog.py` | [Test map](test-map.md) |
| allowed host, redirect, size limit, bzip2, tar+bzip2, `.part`, download cache | [EOS catalog and download](cli/eos-catalog-and-download.md) | `src/pyghl/nn_c2p/eos_catalog.py` | [Generated boundaries](generated-boundaries.md) |
| source tree, editable install, inplace extension | [Build/package/release/CI](build-package-release-ci.md) | `setup.py`, `pyproject.toml` | [Workflows](workflows.md) |
| sdist, wheel, repaired wheel, `package-selected`, `packaged` | [Build/package/release/CI](build-package-release-ci.md) | `MANIFEST.in`, `pyproject.toml`, `setup.py`, workflows | [Generated boundaries](generated-boundaries.md) |
| `cibuildwheel`, manylinux, delocate, `$ORIGIN`, `@loader_path` | [Build/package/release/CI](build-package-release-ci.md) | `setup.py`, `.github/workflows/*.yml` | [Workflows](workflows.md) |
| release, PyPI, Trusted Publishing, wheel workflow artifact | [Build/package/release/CI](build-package-release-ci.md) | `PUBLISHING.md`, `.github/workflows/publish.yml` | [Generated boundaries](generated-boundaries.md) |
| test selection, execution, wheel smoke, upstream tests, manual evidence | [Test map](test-map.md) | Exact assertions/workflow/tests at parent pin | [Workflows](workflows.md) |
| build output, generated file, cache, overwrite, cleanup | [Generated boundaries](generated-boundaries.md) | Exact producer/consumer source | [Workflows](workflows.md) |
| changed file, review fan-out, freshness | [Change-impact map](change-impact.md) | Git state plus exact changed paths | [Index](index.md) |
| contradiction, conflict, safe wording | [Contradictions](contradictions.md) | Competing current repository evidence | [Index](index.md) |
| checker, broken link, orphan, fragment, freshness field | [KB checks](lint/CHECKS.md) | `scripts/check_kb.py`, `tests/test_kb_checks.py` | [KB-only workflow](workflows.md) |

## Path Aliases

| Path/query | Canonical owner | Primary authority | Proof/workflow route |
| --- | --- | --- | --- |
| `csrc/` | [Extension surface and errors](bindings/extension-surface-and-errors.md) | C source | [Change impact](change-impact.md) |
| `src/pyghl/_nn_*`, `src/pyghl/nn.py` | [NN hub](nn/index.md) | Python source | [Source map](source-map.md) |
| `src/pyghl/nn_c2p/` | [CLI hub](cli/index.md) | Python source | [NN hub](nn/index.md) |
| `src/pyghl/nn_c2p/models/*.h5` | [Model/EOS HDF5 lifecycle](nn/model-eos-hdf5-lifecycle.md) | Tracked files plus package metadata | [Generated boundaries](generated-boundaries.md) |
| `tests/test_kb_checks.py` | [KB checks](lint/CHECKS.md) | Exact checker regression assertions | [Change impact](change-impact.md) |
| Other `tests/**` | [Test map](test-map.md) | Exact test source/assertions | [Change impact](change-impact.md) |
| `examples/**` | [Workflows](workflows.md) | Exact example source | [Generated boundaries](generated-boundaries.md) |
| `scripts/train_all.sh` | [CLI command workflows](cli/command-workflows.md) | Exact shell script | [Generated boundaries](generated-boundaries.md) |
| `scripts/check_kb.py` | [KB checks](lint/CHECKS.md) | Exact checker source | [Change impact](change-impact.md) |
| `setup.py`, `pyproject.toml`, `MANIFEST.in` | [Build/package/release/CI](build-package-release-ci.md) | Packaging configuration | [Change impact](change-impact.md) |
| `.github/workflows/kb.yml` | [KB checks](lint/CHECKS.md) | Exact KB workflow YAML | [Change impact](change-impact.md) |
| Other `.github/workflows/*.yml` | [Build/package/release/CI](build-package-release-ci.md) | Exact workflow YAML | [Test map](test-map.md) |
| `.gitignore` | [Generated boundaries](generated-boundaries.md) | Ignore patterns | [Change impact](change-impact.md) |
| `.gitmodules`, `extern/GRHayL` | [GRHayL integration](integration/grhayl-submodule.md) | Parent gitlink and submodule config | [Change impact](change-impact.md) |

## External Ground Truth

No external claim is needed for alias ownership. Exact repository paths and
their canonical owner pages determine these routes.
