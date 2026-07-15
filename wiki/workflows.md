# Repository Workflows and Safety Boundaries

This page routes common operations with prerequisites, mutations, cleanup, and
proof limits. Commands shown in source/docs/workflows are configuration or
instructions until run in a named environment; do not run costly or destructive
operations merely to strengthen documentation.

## Choose the Workflow

| Task | Canonical owner | Primary authority | Artifact/safety route |
| --- | --- | --- | --- |
| KB-only documentation/check | [KB checks](lint/CHECKS.md) | Checker/tests and KB workflow | [Change impact](change-impact.md) |
| Initialize or inspect GRHayL pin | [GRHayL integration](integration/grhayl-submodule.md) | Parent gitlink, `.gitmodules` | [Generated boundaries](generated-boundaries.md) |
| Editable/source/wheel build | [Build/package/release/CI](build-package-release-ci.md) | `setup.py`, metadata, exact command | [Generated boundaries](generated-boundaries.md) |
| Load bindings or EOS | [EOS lifetime/loader](bindings/tabulated-eos-lifetime-and-loader.md) | Python/C source and loader | [Test map](test-map.md) |
| Generate dataset or train | [CLI command workflows](cli/command-workflows.md) | Generator/trainer source | [NN hub](nn/index.md), [generated boundaries](generated-boundaries.md) |
| Append/check/remove model in EOS | [Model/EOS HDF5 lifecycle](nn/model-eos-hdf5-lifecycle.md) | HDF5/CLI source | [Generated boundaries](generated-boundaries.md) |
| Browse/download EOS | [EOS catalog/download](cli/eos-catalog-and-download.md) | Catalog source/tests | [Generated boundaries](generated-boundaries.md) |
| Test proof | [Test map](test-map.md) | Exact assertions/command | Domain owner |
| Release/publish | [Build/package/release/CI](build-package-release-ci.md) | Metadata, PUBLISHING, workflow | Exact hosted artifacts/run |

## KB-Only Workflow

KB work must not configure/build GRHayL, build packages, import product bindings,
download data, train models, open/mutate HDF5, or publish.

```bash
PYTHONDONTWRITEBYTECODE=1 python -m unittest tests.test_kb_checks -v
PYTHONDONTWRITEBYTECODE=1 python scripts/check_kb.py
git diff --check -- AGENTS.md wiki scripts/check_kb.py tests/test_kb_checks.py .github/workflows/kb.yml
```

Prerequisites: Python standard library, Git, repository worktree, and the pinned
GRHayL object for complete delegated-link proof. The checker is read-only. A
missing nested repository/pin object may yield an explicit local proof skip; it
does not authorize validating against a different checkout. These commands prove
local structure/tests when run, not Markdown semantics or hosted CI success.

## GRHayL Source Acquisition and Inspection

Contributor acquisition documented by README:

```bash
git submodule update --init --recursive
```

This may access the network and checkout the superproject-recorded revision,
changing nested state. Never run it over an existing dirty or advanced checkout
without explicit authorization and preservation. For read-only parent claims,
derive the gitlink object from the parent and use `git -C extern/GRHayL
show/grep/cat-file` against it. Do not reset, checkout, update, stage, or clean the
current nested worktree.

## Build and Install Paths

Read [build/package/release/CI](build-package-release-ci.md) before any build.
All source builds can run GRHayL configure/make and mutate a checkout.

| Path | Prerequisites | Mutation/output | Guard and cleanup | Proof after exact run |
| --- | --- | --- | --- | --- |
| Default editable install: `python -m pip install -e .` | Compiler, make, HDF5, build tools, initialized submodule | GRHayL `Makefile`/`build/`, extension/build metadata, environment install | Preserve nested/user state; uninstall environment and remove only proven run-owned files | `compiled`/`linked`; import still separate |
| External checkout: `GRHAYL_DIR=/path/to/GRHayL python -m pip install -e .` | Same, writable configured/source checkout | Mutates external checkout build/config plus environment | Never assume external path disposable | Mode-specific compile/link only |
| Local wheel command from README/PUBLISHING | Same plus wheel frontend | Requested output directory, build tree, GRHayL build; possible package metadata | Use fresh outside-repo output/environment; inspect then discard owned outputs | Named wheel `packaged` only after archive inspection |
| CI wheel | Hosted runner, submodule, platform packages, cibuildwheel | `wheelhouse/*.whl`, repaired macOS wheel, uploaded workflow artifact | Ephemeral runner and artifact retention are platform-owned | YAML is `selected`; hosted result needed for execution |
| Source distribution | PEP 517 backend/frontend | `.tar.gz` plus build metadata | Use disposable output; inspect exact archive | `packaged` for sdist contents only |

Loader behavior branches on the actual `BuildExt.inplace` value. When true, the
extension adds the GRHayL build-library rpath and skips the copy branch. When
false, the build copies recognized `libghl` shared-library names beside the
extension and uses `$ORIGIN` on non-macOS or `@loader_path` on macOS; the macOS
workflow additionally configures delocate repair. Editable and wheel frontend
labels alone do not prove that predicate for a named build; inspect its command
state and output.

## Product Tests

Dependency-qualified root product command:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python -m unittest \
  tests.test_eos_catalog tests.test_nn_c2p_train_cli -v
```

These tests use mocks/temp directories and do not build bindings, use live
network, train, or mutate user HDF5. An import/dependency failure means unrun.
Do not use discovery when recording product-only proof because it also selects
KB tests. Wheel smoke is separate and described in the [test map](test-map.md).

## EOS Loading and Manual Examples

`examples/eos_smoke.py` requires imported bindings and a real compatible EOS
path embedded in the script; source presence is not runnable configuration for
another machine. It reads the table through the binding and prints one pressure.
Run only with an explicitly selected read-only table and close/cleanup according
to [EOS lifetime](bindings/tabulated-eos-lifetime-and-loader.md).

`examples/plot_eps_vs_temp.py` can read tables and either show/save a plot. With
`--output-table`, it copies one EOS then mutates `logenergy` and `logpress` in the
copy. The initial `read_bytes()` copy materializes the full EOS in memory before
overwriting existing output bytes. The resolved-path guard does not detect
distinct hard links to the same inode; a hard-linked output causes the input
inode to be mutated. Verify input/output are different files, back up valuable
data, and budget memory for the full byte copy and loaded arrays.

## Dataset Generation, Training, and Manual Evaluation

These are expensive product operations requiring bindings, EOS data, NumPy,
Torch, h5py, storage, and deliberate output choices.

| Operation | Network | Mutations/outputs | Guard/cleanup |
| --- | --- | --- | --- |
| `python -m pyghl.nn_c2p.nn_c2p_generate_dataset TABLE train|test ...` | No after local table | Binary dataset; default is `nn_training_dataset.bin` for `train` or `nn_test_dataset.bin` for `test`, independent of table name | Explicit `--output` preferred; no automatic cleanup for direct command |
| `pyghl train [EOS] [DATASET] ...` | Yes when EOS omitted | Possible downloaded EOS, temp dataset, log, checkpoints, bundle, NN HDF5, C header, EOS embedded group, installed model | Existing embedded/matching model short-circuits unless force/overwrite; autogenerated dataset removed in `finally`; other outputs persist |
| `scripts/train_all.sh` | No direct download; POSIX `*.h5` expands to matches or remains one literal argument when unmatched | `/tmp` HDF5/bundle/header/log/checkpoints and package-tree installed models | Passes installed-model overwrite; no `set -e`, status aggregation, or cleanup, so later commands continue and final success can mask earlier failure |
| `python -m pyghl.nn_c2p.nn_c2p_test ...` | No | `nn_test*.asc` unless `--output` | Manual comparison output persists; not a unit test |

Training defaults may overwrite named log/output files because writers open
them for write; resume appends the log and checkpoint names can collide. EOS and
installed-model overwrite require their explicit flags, but model output,
bundle, header, log, and dataset paths need caller-side collision review.

## EOS Catalog and Download

`pyghl train` without an EOS invokes interactive catalog discovery/download.
It performs HTTPS requests, may fetch multiple catalog pages, and may write up
to multi-gigabyte archive/decompressed files in current directory. A final path
whose `is_file()` is true is reused as cache, including a symlink to a regular
file outside that directory; a dangling symlink passes the subsequent
`exists()` gate and is replaced by publication. Download source uses `.part`
files, size limits, final replacement, and best-effort partial-file cleanup. It
does not delete final cached EOS tables. Review [catalog trust](cli/eos-catalog-and-download.md)
and ensure disk/network authorization first.

Live remote availability is volatile. Source/tests describe policy; only a named
live run proves availability at that time.

## HDF5 Inspection and Mutation

| Command | Read/write | Guard | Persistent effect |
| --- | --- | --- | --- |
| `pyghl check-eos EOS` | Read | HDF5 open/schema assumptions | Console only |
| `pyghl list-models` | Read | Opens installed model files | Console only |
| `pyghl append EOS [MODEL]` | Write EOS | Existing group requires `--overwrite`; supplied model EOS match unless `--force` | Adds/replaces `grhayl_nn_c2p` in user EOS |
| `pyghl remove-eos-nn EOS` | Write EOS | No-op through CLI if group absent | Deletes embedded group; no backup |
| Model installation during training | Writes package tree | Existing canonical-name model requires overwrite flag | Copies HDF5 under `src/pyghl/nn_c2p/models/` for editable source |

Back up valuable EOS data outside the working path before authorized mutation.
There is no transaction/rollback documented around in-place HDF5 edits. Never
use tracked model or EOS files as scratch data.

## Release and Publication

`PUBLISHING.md` instructs maintainers to update the GRHayL gitlink and version,
run local checks/wheel build, commit/push, confirm Wheels CI, create a GitHub
release, then rely on `.github/workflows/publish.yml`. Push, release creation,
environment/settings changes, and PyPI upload are external state changes and
require separate explicit authority.

Publish workflow text configures release-published trigger, two wheel builds,
artifact upload/download, and `pypa/gh-action-pypi-publish` with OIDC permission.
It does not configure an sdist build/upload. Do not claim a hosted run, artifact
set, trusted-publisher configuration, or PyPI publication without inspecting
those exact external objects.

## External Ground Truth

- [Git submodule documentation](https://git-scm.com/docs/git-submodule) defines
  recursive initialization/update and the superproject-recorded checkout target.
- [PyPA packaging flow](https://packaging.python.org/en/latest/flow/) separates
  source trees, sdists, wheels, uploads, and installed environments.
- [GitHub workflow artifacts](https://docs.github.com/en/actions/concepts/workflows-and-actions/workflow-artifacts)
  defines artifacts as run-produced files persisted or shared after jobs.
- [PyPI Trusted Publishing](https://docs.pypi.org/trusted-publishers/) explains
  OIDC exchange for short-lived publishing credentials; repository YAML alone
  cannot prove PyPI-side publisher configuration.
