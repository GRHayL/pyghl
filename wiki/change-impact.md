# Change-Impact Map

This page maps every repository path family to canonical owners and proportional
review. It routes review; changed source/config/tests and parent-pinned objects
remain authority.

## Use This Map

Collect committed, staged, unstaged, untracked, and submodule changes before
review. Match each path below, reopen exact sources, update owner leaves first,
then reconcile tests, artifacts, workflows, and routers named by the row. One
change may match several rows.

| Changed path family | Canonical owner | Required review fan-out | Proportional proof |
| --- | --- | --- | --- |
| `AGENTS.md`, `wiki/**` | [Index](index.md) and changed owner page | Routes, owner uniqueness, hop budget, links, evidence states | KB checker/tests plus semantic source trace |
| `scripts/check_kb.py`, `tests/test_kb_checks.py`, `.github/workflows/kb.yml` | [KB checks](lint/CHECKS.md) | Index policy, diagnostics, pin correctness, workflow parity | Checker regression suite and full graph check |
| `csrc/pyghl_module.c` | [Extension surface/errors](bindings/extension-surface-and-errors.md) or [EOS lifetime](bindings/tabulated-eos-lifetime-and-loader.md) | Public API map, GRHayL integration, tests, build ABI | Registration inventory, parent-pin trace, targeted wrapper execution when authorized |
| `src/pyghl/__init__.py` | [Public API map](public-api-map.md) | Binding loader owner, conditional exports, CLI/NN imports | Import/error-path tests in named environments |
| `src/pyghl/eos.py` | [EOS lifetime/loader](bindings/tabulated-eos-lifetime-and-loader.md) | API map, workflows, tests | Path-resolution and initialization tests |
| `src/pyghl/nn.py` | [NN hub](nn/index.md) | Public facade, dataset/inference/HDF5 owners, contradictions | Targeted facade/import tests |
| `src/pyghl/_nn_common.py` | [Training/checkpoints](nn/training-and-checkpoints.md) | Inference/export, datasets, tests | Transform/training/inference assertions |
| `src/pyghl/_nn_dataset.py` | [Dataset/generation](nn/dataset-and-generation.md) | Generated boundaries, training, tests | Producer/reader fixture tests |
| `src/pyghl/_nn_train.py` | [Training/checkpoints](nn/training-and-checkpoints.md) | Inference/export, HDF5, CLI, generated boundaries, tests | Targeted training/export/checkpoint tests; no routine training |
| `src/pyghl/_nn_infer.py` | [Inference/export](nn/inference-and-export.md) | Training bundle contract, API map, tests | Bundle/decode/inference fixture tests |
| `src/pyghl/_nn_hdf5.py` | [Model/EOS HDF5 lifecycle](nn/model-eos-hdf5-lifecycle.md) | CLI mutation, generated boundaries, tracked models, pinned loader, tests | Temp-file schema/hash/mutation tests and pin comparison |
| `src/pyghl/cli.py` | [CLI command workflows](cli/command-workflows.md) | Public API map, catalog aliases, tests, README claims | Parser/dispatch/exit tests |
| `src/pyghl/nn_c2p/eos_catalog.py` | [EOS catalog/download](cli/eos-catalog-and-download.md) | Test map, workflows, generated downloads/cache | Mock/temp security tests; live network separate |
| `src/pyghl/nn_c2p/__init__.py` | [NN hub](nn/index.md) | CLI namespace/routes and public API map | Import/namespace inspection |
| `src/pyghl/nn_c2p/nn_c2p_generate_dataset.py`, `common.py` | [Dataset/generation](nn/dataset-and-generation.md) | CLI workflow, binding prerequisites, artifacts, training/test consumers | Producer/reader fixture and targeted helper tests |
| `src/pyghl/nn_c2p/nn_c2p_train.py` | [Training/checkpoints](nn/training-and-checkpoints.md) | CLI command workflow, catalog selection, HDF5 mutation, generated boundaries | Parser/orchestration tests; training execution only when authorized |
| `src/pyghl/nn_c2p/nn_c2p_test.py`, `header_to_hdf5.py` | [Inference/export](nn/inference-and-export.md) | CLI/module routes, HDF5 schema, manual outputs | Targeted conversion tests; manual evaluation separate |
| `src/pyghl/nn_c2p/append_eos_file.py`, `check_eos.py`, `list_installed_models.py`, `remove_eos_nn.py` | [Model/EOS HDF5 lifecycle](nn/model-eos-hdf5-lifecycle.md) | CLI command workflow, mutation guards, generated boundaries, tests | Temp-file lifecycle/parser tests |
| `src/pyghl/nn_c2p/models/*.h5` | [Model/EOS HDF5 lifecycle](nn/model-eos-hdf5-lifecycle.md) | Packaging selection, generated boundaries, release proof, pinned loader | Per-file read-only inspection; loader/numerical/package proof separate |
| `src/pyghl/py.typed` | [Build/package/release/CI](build-package-release-ci.md) | Package selection and inspected wheel/sdist | Inspect exact artifact and typing consumer as needed |
| `tests/test_eos_catalog.py`, `tests/test_nn_c2p_train_cli.py` | [Test map](test-map.md) | Corresponding CLI/NN owner and workflow selection | Run exact product command if dependencies available |
| Other `tests/**` | [Test map](test-map.md) | Matching owner, checker/workflow selection | Run exact named suite; presence is not pass |
| `examples/**` | [Workflows](workflows.md) | Matching binding/NN/CLI owner and generated boundaries | Read source; manual execution only when authorized with safe inputs |
| `scripts/train_all.sh` | [CLI command workflows](cli/command-workflows.md) | Training, installed-model mutation, `/tmp` artifacts, cleanup | Read-only shell review; deliberate disposable execution only |
| `setup.py` | [Build/package/release/CI](build-package-release-ci.md) | GRHayL integration, source/API maps, generated boundaries, wheels | Controlled build/package/load checks by mode |
| `pyproject.toml` | [Build/package/release/CI](build-package-release-ci.md) | Version/dependencies/scripts/package data, API map, contradictions, release | Inspect generated metadata and exact artifacts |
| `MANIFEST.in` | [Build/package/release/CI](build-package-release-ci.md) | Sdist selection, wheel-from-sdist risks, generated boundaries | Inspect exact named sdist; build wheel from sdist if claim requires |
| `.github/workflows/wheels.yml` | [Build/package/release/CI](build-package-release-ci.md) | Platform/Python matrix, smoke proof, artifacts, test map | YAML/text review; exact hosted run for success |
| `.github/workflows/publish.yml` | [Build/package/release/CI](build-package-release-ci.md) | Release trigger, permissions, wheel artifacts, PyPI, PUBLISHING | YAML/text review; exact hosted run and published artifacts for release |
| Other `.github/**` | [Build/package/release/CI](build-package-release-ci.md) | Workflows/test selection/security/artifacts | Exact config review and hosted evidence if claimed |
| `README.md` | [Index](index.md) | User claims against source/API/workflows; contradictions | Triangulate each changed claim with primary authority |
| `PUBLISHING.md` | [Build/package/release/CI](build-package-release-ci.md) | Version, gitlink, workflows, artifacts, trusted publishing | Triangulate checklist with metadata/YAML and hosted evidence |
| `.gitignore` | [Generated boundaries](generated-boundaries.md) | Every producer location, cleanup, state-preservation rules | Compare exact patterns with build/training/download/test outputs |
| `.gitmodules` | [GRHayL integration](integration/grhayl-submodule.md) | Pin acquisition, external checkout/build, workflows, source map | Parent gitlink/object checks; no checkout reset/update |
| `extern/GRHayL` gitlink | [GRHayL integration](integration/grhayl-submodule.md) | C wrappers, EOS/HDF5 loader, build/link inputs, upstream routes/tests | Inspect new parent-pinned objects; rebuild/test only if authorized |
| `extern/GRHayL/**` working tree | [Nested GRHayL instructions](../extern/GRHayL/AGENTS.md) | Parent KB only if integration behavior or parent gitlink changes | Preserve nested state; nested repository rules apply |

## Cross-Cutting Triggers

- Any producer/output path change reviews [generated boundaries](generated-boundaries.md)
  and `.gitignore`; generated files never become proof merely by existing.
- Any parser/export/registration change reviews [public API map](public-api-map.md)
  and [catalog](catalog.md).
- Any source behavior change reviews [test map](test-map.md); missing proof stays a
  gap, not a contradiction.
- Any build/package/workflow change keeps `selected`, `package-selected`,
  `packaged`, `compiled`, `linked`, `imported`, hosted execution, and publication
  distinct.
- Any user-visible competing claim is triangulated for [contradictions](contradictions.md).
  Checkout drift, narrower matrices, and evidence gaps do not enter that table.

## External Ground Truth

- [Git submodule documentation](https://git-scm.com/docs/git-submodule) explains
  that the superproject-recorded submodule commit can differ from the checked-out
  submodule commit; use this distinction when reviewing `.gitmodules` or the
  `extern/GRHayL` gitlink.
