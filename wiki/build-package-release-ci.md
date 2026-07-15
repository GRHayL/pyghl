# Build, Package, Release, and CI Boundaries

This page owns pyghl build selection, extension/link/load modes, package formats,
wheel workflows, and release configuration. `setup.py`, metadata, manifests,
workflow YAML, and PUBLISHING are configuration authority; only exact artifacts,
environments, hosted runs, and releases support stronger evidence states.

## Build Inputs and Selection

`pyproject.toml` selects setuptools' PEP 517 backend and declares project name,
version, Python `>=3.9`, normal dependencies (`numpy`, `torch`, `h5py`), console
script, src layout, explicit shared-library/model package-data globs, and
`include-package-data = true`. `MANIFEST.in` separately selects
`src/**/*.typed`, including `py.typed`; together these configurations select the
typing marker for source and wheel distribution inputs, while exact sdist/wheel
inspection remains required to establish artifact contents. `setup.py` selects
one C99 extension from `csrc/pyghl_module.c`, parent-pinned/default or external
GRHayL headers/library, custom GRHayL configure/make, runtime rpaths, and
non-inplace shared-library copying.

| Mode/input | Selection and flow | Mutation/output | Current evidence |
| --- | --- | --- | --- |
| Clean default submodule | No `GRHAYL_DIR`; use `extern/GRHayL`. If Makefile or build dir is absent, call configure with `--prefix=.` plus `GRHAYL_CONFIGURE_ARGS`; then always `make ... grhayl` | Mutates nested Makefile/build and build frontend outputs | `implemented`; no build run |
| Preconfigured default submodule | Existing Makefile and build dir skip configure; make still runs | Updates nested build artifacts | `implemented`; preconfiguration does not prove compatible libs |
| External checkout | `GRHAYL_DIR` resolves an alternate root; same configure/make/link logic | Mutates user-owned external checkout | `implemented`; external state/compatibility unproved |
| `BuildExt.inplace` true | `build_extension` adds loader-local rpath plus absolute GRHayL build-lib rpath; `run` skips the copy branch | Named build output/environment | `implemented`; an editable frontend label alone does not prove the predicate or compile/link/import |
| `BuildExt.inplace` false | Build GRHayL, compile/link extension, copy recognized `libghl` names beside extension, rewrite macOS load paths | Named package/build tree; may become wheel input | `implemented`/`package-selected`; a wheel frontend label alone does not prove the predicate or artifact contents/load |

Include roots are `GRHAYL_ROOT/GRHayL/include` and
`GRHAYL_ROOT/Unit_Tests/data_gen`; library root is
`GRHAYL_ROOT/build/lib`, and link library is `ghl`. These exact configuration
paths do not establish that headers/libs exist or match until inspected/built.
Parent-pinned upstream authority and safe inspection live in [GRHayL
integration](integration/grhayl-submodule.md).

## Loader Boundaries

| Platform/mode | Configured path behavior | Required stronger proof |
| --- | --- | --- |
| Non-macOS `inplace` false | Extension gets `$ORIGIN`; copied `libghl*.so` is selected beside it | Establish actual predicate; inspect named build or wheel files/dependencies, then import exact install |
| Non-macOS `inplace` true | `$ORIGIN` plus absolute `BUILD_LIB_DIR` rpath | Establish actual predicate; inspect extension dynamic section and import exact environment |
| macOS `inplace` false | `@loader_path`; copied `.dylib` IDs and extension references rewritten with `install_name_tool` | Establish actual predicate; inspect named build or repaired wheel Mach-O references and exact install |
| macOS workflow wheel | Above plus configured delocate repair requiring target architecture | Exact repair log, repaired wheel inspection, installed import |
| Binding facade | `src/pyghl/__init__.py` catches extension `ImportError`, exports NN regardless, conditionally exports binding/eos names, and `require_bindings()` rethrows guidance | Exact source is `exported`; only named environment import is `imported` |

An extension being compiled does not prove it linked the intended library;
linking does not prove runtime resolution; runtime import does not prove wrapper
or numerical behavior.

## Distribution Selection

| Package surface | Selecting configuration | Evidence limit |
| --- | --- | --- |
| Project metadata and console entry | `pyproject.toml` `[project]` and `[project.scripts]` | Selected metadata; inspect exact sdist/wheel `PKG-INFO`/`METADATA` and entry points |
| Python packages | setuptools src-layout discovery | `package-selected`, not archived/importable proof |
| Shared libs and four model HDF5 files | explicit `pyproject.toml` package-data patterns | `package-selected`; glob/matched source presence is not wheel contents |
| `py.typed` | tracked marker selected by `MANIFEST.in` plus `include-package-data = true` | `package-selected`; inspect an exact sdist/wheel before claiming artifact contents |
| README/C/Python/typed/model source-distribution inputs | `MANIFEST.in` recursive/include rules; no rule selects `extern/GRHayL` | Source-distribution selection only; repository selection does not establish that the default GRHayL path needed by `setup.py` exists in an sdist. Inspect the exact `.tar.gz` and name any required external `GRHAYL_DIR` checkout before claiming a wheel can be built from it |
| CPython extension | `setup.py` `Extension` plus `BuildExt` | Build selection; exact wheel/platform/ABI inspection required |

No wheel or sdist is present in the tracked repository inventory. No exact
archive was built or inspected for this page, so there is no current `packaged`
claim. Four tracked model files remain `source-present` and `package-selected`
only; see [HDF5 lifecycle](nn/model-eos-hdf5-lifecycle.md).

## Wheel CI Configuration

`.github/workflows/wheels.yml` configures pull-request, push-to-main, and manual
triggers. `.github/workflows/publish.yml` configures the `release` event with
type `published`. Both configure the same two matrix entries and cibuildwheel environment:

| Matrix/config | Linux x86_64 | macOS arm64 |
| --- | --- | --- |
| Runner | `ubuntu-26.04` | `macos-14` |
| cibuildwheel architecture | `x86_64` | `arm64` |
| CPython selection | `cp310-* cp311-* cp312-* cp313-*` | Same |
| Skips | PyPy and musllinux | Same generic skip |
| System preparation | HDF5/pkg-config/zlib/libaec through dnf/yum | Homebrew HDF5/pkg-config; install delocate before build |
| GRHayL configure args | HDF5 include/lib paths | Homebrew HDF5 prefix; deployment target 14.0 |
| Repair | cibuildwheel platform default/config | `delocate-wheel --require-archs ...` |
| Smoke | Import `pyghl`, call `require_bindings()` | Same |
| Output | Upload `wheelhouse/*.whl` as matrix-named artifact | Same |

Both checkout steps configure `submodules: recursive`, acquiring source objects
needed for the build. This is source checkout, not wrapper/runtime proof. Matrix
text is `selected`, not compiled/imported/hosted evidence. The metadata Python
floor (`>=3.9`) and CPython 3.10–3.13 wheel selection answer different questions:
metadata permits candidate source installs while these workflows select a
narrower wheel matrix; neither implies platform support outside exact artifacts.

Workflow YAML presence also does not prove GitHub accepted syntax, actions still
resolve, system packages install, jobs ran, smoke passed, or artifacts exist.

## Release and Publish Chain

`PUBLISHING.md` describes this maintainer sequence:

1. Update/review the GRHayL gitlink.
2. Set version in `pyproject.toml` and run local checks/wheel build.
3. Commit/push; wait for Wheels on main and inspect/test artifacts when relevant.
4. Create `vX.Y.Z` GitHub release targeting matching source/version.
5. Release publication triggers `.github/workflows/publish.yml`.

Publish workflow then configures two wheel build jobs, uploads matrix artifacts,
downloads and merges them under `dist`, and invokes
`pypa/gh-action-pypi-publish@release/v1` in environment `pypi`. Workflow-level
permissions grant contents read and OIDC `id-token: write`; publish job depends
on both wheel jobs. Repository text does not prove the PyPI trusted publisher or
GitHub environment/settings are configured.

The publish workflow does not configure source-distribution creation/upload.
Therefore safe wording is “the repository configures wheel publication on a
published GitHub release,” not “every release contains an sdist and wheels” or
“version X is published.” An exact hosted run plus immutable release/PyPI
artifact inspection is needed for `release-published`.

## Package and Release Inspection

Run only for an authorized specific claim, in disposable directories:

- For sdist: record exact filename; inspect tar members, `pyproject.toml`,
  `PKG-INFO`, C source, Python source, typing marker, and expected model inputs.
- For wheel: record filename/tags; inspect ZIP members, `METADATA`, `RECORD`,
  entry points, extension, copied/repaired shared libs, typing marker, and model
  files. Do not execute an archive directly.
- For runtime: install the exact wheel in a fresh environment without silently
  falling back to source; import `pyghl`, call `require_bindings()`, and keep
  behavioral tests separate.
- For hosted CI/release: record exact commit/run/job/artifact/release/version and
  compare artifact identities across upload/download/publication.

Artifact locations, ownership, and cleanup are in [generated boundaries](generated-boundaries.md).
Operational commands and mutation gates are in [workflows](workflows.md).

## Known Proof Gaps

- No inspected sdist, wheel, repaired wheel, installed environment, link report,
  hosted run, GitHub release, or PyPI release was used for this page.
- No root test directly asserts archive contents, link/load behavior, metadata
  parity, matrix behavior, or publication.
- Wheel smoke does not run product tests or numerical APIs.
- Metadata license expression is source-present; no root license text file is
  tracked. This is an inventory fact, not by itself a conflicting repository
  claim.
- Optional-install guidance conflict is recorded in [contradictions](contradictions.md).

## External Ground Truth

- [Setuptools data-file configuration](https://setuptools.pypa.io/en/latest/userguide/datafiles.html)
  defines how `MANIFEST.in`, `include-package-data`, and explicit package-data
  select sdist and wheel inputs; exact produced archives still require inspection.
- [PyPA package formats](https://packaging.python.org/en/latest/discussions/package-formats/)
  distinguishes source distributions and wheels, explains compiled-extension
  wheel specificity, and documents archive inspection.
- [PyPA packaging flow](https://packaging.python.org/en/latest/flow/) distinguishes
  source tree/configuration, sdist, wheel, upload, and installation stages.
- [Source distribution specification](https://packaging.python.org/en/latest/specifications/source-distribution-format/)
  defines the current sdist archive requirements.
- [actions/checkout documentation](https://github.com/actions/checkout) defines
  `submodules: recursive` as recursive submodule checkout; this proves checkout
  configuration only.
- [GitHub Actions workflow syntax](https://docs.github.com/en/actions/reference/workflows-and-actions/workflow-syntax)
  defines triggers, matrices, jobs, steps, permissions, and filters; repository
  text still does not prove platform acceptance or execution.
- [GitHub workflow artifacts](https://docs.github.com/en/actions/concepts/workflows-and-actions/workflow-artifacts)
  explains upload/download persistence and job sharing.
- [PyPI Trusted Publishing](https://docs.pypi.org/trusted-publishers/) explains
  OIDC-based short-lived publishing credentials and required PyPI-side trust.
- [PyPI file-reuse help](https://pypi.org/help/#file-name-reuse) states that a
  distribution filename cannot be reused, including after deletion.
