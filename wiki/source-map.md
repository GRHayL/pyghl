# Source Map

This map routes every tracked pyghl path family to one canonical KB owner. It
describes ownership and evidence location, not runtime success; exact source,
configuration, tests, workflows, artifacts, and parent-pinned GRHayL objects
remain authority for their respective claims.

## Repository path ownership

| Tracked path or family | Owned behavior | Canonical owner | Primary authority | Proof/workflow route |
| --- | --- | --- | --- | --- |
| `AGENTS.md` | Root scope, safety, and first-hop routing | [KB index](index.md) | `AGENTS.md` | [KB checks](lint/CHECKS.md) |
| `wiki/**` | Knowledge graph, owner leaves, and routing policy | [KB index](index.md) | `wiki/index.md`, owning page | [KB checks](lint/CHECKS.md) |
| `scripts/check_kb.py`, `tests/test_kb_checks.py`, `.github/workflows/kb.yml` | KB structural checker, regression assertions, and hosted selection | [KB checks](lint/CHECKS.md) | exact checker/test/workflow files | [Change impact](change-impact.md) |
| `csrc/pyghl_module.c` | CPython types, functions, constants, parsing, error translation, and EOS ownership | [Extension surface and errors](bindings/extension-surface-and-errors.md) | exact C registration tables and wrapper symbols | [Bindings hub](bindings/index.md), [test map](test-map.md) |
| `src/pyghl/__init__.py` | Conditional extension import and top-level Python export contract | [Public API map](public-api-map.md) | import block, `require_bindings`, and `__all__` | [Tabulated EOS loader](bindings/tabulated-eos-lifetime-and-loader.md) |
| `src/pyghl/eos.py` | Tabulated-EOS path resolution and Python initializer facade | [Tabulated EOS loader](bindings/tabulated-eos-lifetime-and-loader.md) | `_resolve_table_path`, `initialize_tabulated_eos_functions_and_params` | [Bindings hub](bindings/index.md) |
| `src/pyghl/nn.py` | Public pure-Python NN facade | [NN hub](nn/index.md) | class/function definitions and deferred imports | [Public API map](public-api-map.md) |
| `src/pyghl/_nn_common.py` | Model, scaling, transform, and shared NN contracts | [Training and checkpoints](nn/training-and-checkpoints.md) | exact classes/functions | [Inference and export](nn/inference-and-export.md), [NN hub](nn/index.md) |
| `src/pyghl/_nn_dataset.py` | Binary dataset reading and validation | [Dataset and generation](nn/dataset-and-generation.md) | exact reader constants/functions | [Generated boundaries](generated-boundaries.md) |
| `src/pyghl/_nn_train.py` | Training, checkpoints, bundles, and C-header export | [Training and checkpoints](nn/training-and-checkpoints.md) | exact training/export functions | [Inference and export](nn/inference-and-export.md) |
| `src/pyghl/_nn_infer.py` | Python inference and output interpretation | [Inference and export](nn/inference-and-export.md) | exact inference functions | [Test map](test-map.md) |
| `src/pyghl/_nn_hdf5.py` | Standalone/embedded NN HDF5 schema, identity, install, and mutation | [Model/EOS HDF5 lifecycle](nn/model-eos-hdf5-lifecycle.md) | format constants and HDF5 producer/consumer functions | [Generated boundaries](generated-boundaries.md) |
| `src/pyghl/nn_c2p/__init__.py` | NN command-package namespace | [NN hub](nn/index.md) | package source | [CLI hub](cli/index.md) |
| `src/pyghl/nn_c2p/nn_c2p_generate_dataset.py`, `common.py` | C2P dataset generation/orchestration and shared flat-metric helper | [Dataset and generation](nn/dataset-and-generation.md) | parser, generator, writers, and `set_flat_metric` | [Command workflows](cli/command-workflows.md) |
| `src/pyghl/nn_c2p/nn_c2p_train.py` | End-to-end training orchestration | [Training and checkpoints](nn/training-and-checkpoints.md) | parser and orchestration functions | [Command workflows](cli/command-workflows.md) |
| `src/pyghl/nn_c2p/nn_c2p_test.py`, `header_to_hdf5.py` | Evaluation and conversion | [Inference and export](nn/inference-and-export.md) | parser and conversion/evaluation functions | [Generated boundaries](generated-boundaries.md) |
| `src/pyghl/nn_c2p/append_eos_file.py`, `check_eos.py`, `list_installed_models.py`, `remove_eos_nn.py` | EOS/model inspection, matching, installation, embedding, listing, and removal | [Model/EOS HDF5 lifecycle](nn/model-eos-hdf5-lifecycle.md) | exact command and helper functions | [Command workflows](cli/command-workflows.md) |
| `src/pyghl/nn_c2p/eos_catalog.py` | Remote catalog discovery, selection, download, archive extraction, and cache behavior | [EOS catalog and download](cli/eos-catalog-and-download.md) | exact catalog/download functions | [Test map](test-map.md) |
| `src/pyghl/nn_c2p/models/*.h5` | Tracked installed-model artifacts | [Model/EOS HDF5 lifecycle](nn/model-eos-hdf5-lifecycle.md) | actual artifact plus selection metadata | [Generated boundaries](generated-boundaries.md) |
| `src/pyghl/cli.py` | Installed command spelling and first-level dispatch | [Command workflows](cli/command-workflows.md) | `build_parser`, `main` | [CLI hub](cli/index.md) |
| `src/pyghl/py.typed` | Typing marker selected as package data | [Build/package/release/CI](build-package-release-ci.md) | file presence and package metadata | [Public API map](public-api-map.md) |
| `tests/test_eos_catalog.py`, `tests/test_nn_c2p_train_cli.py`, `tests/__init__.py` | Direct repository test assertions and test-package marker | [Test map](test-map.md) | exact tests and assertions | [Workflows](workflows.md) |
| `examples/*.py` | Operational examples; not direct test evidence | [Workflows](workflows.md) | exact scripts | relevant bindings/NN owner leaf |
| `scripts/train_all.sh` | Batch training CLI orchestration | [Command workflows](cli/command-workflows.md) | exact shell script | [Training and checkpoints](nn/training-and-checkpoints.md), [workflows](workflows.md) |
| `pyproject.toml` | Project metadata, dependencies, console script, package discovery, and package-data selection | [Build/package/release/CI](build-package-release-ci.md) | exact TOML tables | [Public API map](public-api-map.md) |
| `setup.py` | GRHayL selection/configure/build, extension link, rpath, shared-library copy, and macOS rewrite | [Build/package/release/CI](build-package-release-ci.md) | exact build command class and extension declaration | [GRHayL integration](integration/grhayl-submodule.md) |
| `MANIFEST.in` | Source-distribution selection | [Build/package/release/CI](build-package-release-ci.md) | exact manifest rules | [Generated boundaries](generated-boundaries.md) |
| `.github/workflows/wheels.yml`, `.github/workflows/publish.yml` | Wheel/release job configuration | [Build/package/release/CI](build-package-release-ci.md) | exact workflow YAML | [Test map](test-map.md), [workflows](workflows.md) |
| `.gitmodules` | Submodule name/path/URL/branch hints | [GRHayL integration](integration/grhayl-submodule.md) | `.gitmodules` plus parent gitlink | [Change impact](change-impact.md) |
| `extern/GRHayL` gitlink and checkout | Parent-selected GRHayL revision and independent upstream repository | [GRHayL integration](integration/grhayl-submodule.md) | parent tree gitlink, then objects in nested object database | nested `extern/GRHayL/AGENTS.md` at parent pin |
| `.gitignore` | Generated and ignored workspace boundaries | [Generated boundaries](generated-boundaries.md) | exact ignore rules | [Change impact](change-impact.md) |
| `README.md` | User onboarding and examples; supporting documentation only | [KB index](index.md) | exact implementation/configuration for behavioral claims | owning leaf and [contradictions](contradictions.md) |
| `PUBLISHING.md` | Maintainer publishing checklist; supporting release documentation | [Build/package/release/CI](build-package-release-ci.md) | metadata, workflow, and inspected release artifact | [Workflows](workflows.md) |

## Boundary routes

| Query | Canonical owner | Primary authority | Proof/workflow route |
| --- | --- | --- | --- |
| A Python name is absent after import | [Public API map](public-api-map.md) | `src/pyghl/__init__.py` | [Tabulated EOS loader](bindings/tabulated-eos-lifetime-and-loader.md) for loader failures |
| A wrapper field, return, mutation, or exception changes | [Extension surface and errors](bindings/extension-surface-and-errors.md) | `csrc/pyghl_module.c` | [Test map](test-map.md) |
| A GRHayL declaration or algorithm changes | [GRHayL integration](integration/grhayl-submodule.md), then pinned nested instructions | parent gitlink and parent-pinned upstream object | upstream tests are upstream proof only |
| Build finds a different GRHayL tree | [Build/package/release/CI](build-package-release-ci.md) | `setup.py` and environment | [GRHayL integration](integration/grhayl-submodule.md) |
| Model or EOS HDF5 is created or mutated | [Model/EOS HDF5 lifecycle](nn/model-eos-hdf5-lifecycle.md) | exact producer/consumer source and actual artifact | [Generated boundaries](generated-boundaries.md) |
| CLI command spelling differs from delegated behavior | [Command workflows](cli/command-workflows.md) | `src/pyghl/cli.py`, then delegated command module | [CLI hub](cli/index.md) |

## Evidence limits

Path presence is `source-present`, not execution. Registration tables can prove
`registered`; facade imports can prove `exported`; metadata can prove
`selected` or `package-selected`. Only inspection of a named built artifact can
prove `packaged`, and only a named environment/run can prove `imported` or
`executed`. See [KB index](index.md) for full evidence vocabulary.
