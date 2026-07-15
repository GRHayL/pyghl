# Test and Proof Map

This page routes proof questions and records direct repository test coverage and
explicit gaps. Exact assertions are authority for test intent; a test file or
workflow selection is not evidence that a test executed or passed.

## Evidence Layers

| Proof question | Canonical owner | Primary authority | Stronger evidence required |
| --- | --- | --- | --- |
| Is behavior asserted in product tests? | This page, then domain leaf | `tests/test_eos_catalog.py`, `tests/test_nn_c2p_train_cli.py` | Named command result in a stated environment |
| Is KB structure checked? | [KB checks](lint/CHECKS.md) | `tests/test_kb_checks.py`, `scripts/check_kb.py` | Local command result; hosted result is separate |
| Is a test selected by a workflow? | [Build/package/release/CI](build-package-release-ci.md) | Exact workflow command/environment | Hosted workflow run for the exact commit |
| Does a wheel import? | [Build/package/release/CI](build-package-release-ci.md) | `CIBW_TEST_COMMAND` in wheel workflow | Exact hosted/local wheel environment result |
| Does a built package contain a file? | [Generated boundaries](generated-boundaries.md) | Contents of exact named wheel/sdist | Artifact inspection; metadata is insufficient |
| Does a model/EOS satisfy a schema? | [Model/EOS HDF5 lifecycle](nn/model-eos-hdf5-lifecycle.md) | Exact named file and inspected fields | Loader execution and numerical checks are separate |
| Does GRHayL behavior work? | [GRHayL integration](integration/grhayl-submodule.md) | Tests/source at parent-recorded revision | Does not prove pyghl wrapper behavior |
| Does an example or module tool work? | [Workflows](workflows.md) | Exact script source | Manual execution with named prerequisites; not a unit test |

Evidence remains separated:

- `selected`: a command, matrix, or metadata names work.
- `executed`: that exact command ran with named prerequisites.
- Wheel smoke is a package/import seam, not product behavioral coverage.
- Upstream tests cover upstream code, not CPython parsing, lifetimes, packaging,
  conditional exports, or Python orchestration.
- Manual scripts may exercise integration while creating or mutating user files;
  they are not automatically repeatable tests.

## Product Test Groups

### EOS catalog and download

`tests/test_eos_catalog.py` contains 22 `unittest` methods. Tests use mocked
openers, in-memory responses/archives, fake curses screens, and temporary
directories; they do not prove current live remote availability.

| Behavior group | What assertions cover | Exact authority | Owner |
| --- | --- | --- | --- |
| Progress rendering | Percentage/bytes/speed/ETA with a content length; bytes/speed/elapsed without one | `_DownloadProgress` assertions | [EOS catalog/download](cli/eos-catalog-and-download.md) |
| Catalog discovery and parsing | Microphysics family discovery/deduplication, row/family pairing, APR tar+bzip2 links, SRO description/link pairing, recursive nested pages | Parser/fetch assertions and fixture HTML | [EOS catalog/download](cli/eos-catalog-and-download.md) |
| Trust and limits | Rejection of an unallowed download host and oversized HTML; network timeout passed to opener | Error and call assertions | [EOS catalog/download](cli/eos-catalog-and-download.md) |
| Download lifecycle | bzip2 and tar+bzip2 extraction, response-size progress, decompressed cache reuse, directory collision, final destination bytes/publication, partial cleanup after invalid archive | Temporary-directory contents and bytes; assertions do not prove atomicity | [EOS catalog/download](cli/eos-catalog-and-download.md) |
| Picker behavior | Text filtering including lowercase `k`, category-before-table selection, Stockholm warning rendering | Fake screen key/render assertions | [EOS catalog/download](cli/eos-catalog-and-download.md) |
| Top-level orchestration | Fetch, select, and download call order | Mock call assertions | [CLI command workflows](cli/command-workflows.md) |

These tests do not directly exercise real HTTP/TLS/DNS, real redirects, full
configured compressed/decompressed limits, malicious tar variants beyond the
constructed member, actual curses terminals, HDF5 validity, or live catalog
contents.

### Training CLI EOS selection

`tests/test_nn_c2p_train_cli.py` contains five `unittest` methods.

| Behavior group | What assertions cover | Exact authority | Owner |
| --- | --- | --- | --- |
| Parser | EOS argument is optional; explicit path remains accepted | Parser result assertions | [CLI command workflows](cli/command-workflows.md) |
| Resolution | Missing path delegates to remote chooser; explicit path skips chooser | Mock result/call assertions | [EOS catalog/download](cli/eos-catalog-and-download.md) |
| Cancellation | `KeyboardInterrupt` from selection returns 130 and prints cancellation | Return/stderr assertions | [CLI command workflows](cli/command-workflows.md) |

These tests do not invoke bindings, generation, training, checkpointing,
artifact export, model matching, EOS mutation, installed-model registration, or
cleanup after a failure inside those later stages.

## Configured Wheel Smoke

Both `.github/workflows/wheels.yml` and `.github/workflows/publish.yml` configure
`cibuildwheel` for Linux x86_64 and macOS arm64 with CPython 3.10–3.13 selection.
Their `CIBW_TEST_COMMAND` imports `pyghl`, calls `pyghl.require_bindings()`, and
prints a marker. Workflow text therefore proves `selected` smoke behavior only.
It does not prove a hosted run, numerical behavior, CLI behavior, shared-library
contents, all metadata-supported Python versions, or root product-test execution.

No repository workflow visibly selects `tests/test_eos_catalog.py` or
`tests/test_nn_c2p_train_cli.py`. The dedicated KB workflow selects only checker
tests and the checker; see [KB checks](lint/CHECKS.md).

## Manual and Upstream Evidence

| Evidence | What it can show after execution | What it cannot show by presence |
| --- | --- | --- |
| `examples/eos_smoke.py` | One binding/EOS resolution/interpolation path with a real table | Broad wrapper or lifecycle coverage |
| `examples/nn_c2p_generate_dataset.py` | Dataset generation wrapper under real bindings | Reader/training correctness or cleanup generally |
| `examples/nn_c2p_train.py` | CLI training module path | Reproducibility, model validity, or installed CLI dispatch |
| `examples/nn_c2p_test.py` | Manual C2P comparison tool and ASCII output | Unit-test pass or Python/C parity by source presence |
| `examples/plot_eps_vs_temp.py` | Read/plot or copied-EOS rewrite behavior | EOS physical correctness or safe mutation for arbitrary tables |
| `scripts/train_all.sh` | Batch invocation when deliberately run | Success for any table; it overwrites installed models and creates `/tmp` outputs |
| Parent-pinned `extern/GRHayL/Unit_Tests/` | Upstream behavior for the pinned revision when run | CPython wrapper, Python facade, packaging, or CLI behavior |

## Direct Coverage Gaps

No comparable root product tests were found for:

- C wrapper type construction, get/set parsing, validation, error translation,
  constants/functions, or representative numerical calls;
- `TabulatedEOS` initialization, method-function prerequisites, NN load, close,
  deallocation, repeated close, and post-close failures;
- `src/pyghl/__init__.py` conditional exports, `require_bindings()` diagnostics,
  editable shared-library discovery, and regular-install loader paths;
- dataset header/record rejection and producer/reader agreement;
- feature transforms, scaling, filtering, deterministic split, training, early
  stopping, checkpoint/resume, bundle load, inference decode, and C export parity;
- standalone/embedded HDF5 schema, canonical hashing, append/overwrite/remove,
  installed-model matching, tracked model compatibility, and pinned GRHayL load;
- top-level dispatch for append/check/list/remove/version and CLI mutation guards;
- sdist contents, wheel contents, repaired dependency contents, metadata tags,
  release artifact set, and published package;
- Linux/macOS build/load behavior outside the configured wheel smoke;
- live catalog/download behavior and large-file limits.

A gap is not a contradiction and not evidence of failure. Close one only with an
exact assertion/artifact/run and update the owning leaf before this summary.

## Commands and Proof Limits

Dependency-qualified product tests:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python -m unittest \
  tests.test_eos_catalog tests.test_nn_c2p_train_cli -v
```

KB-only checks:

```bash
PYTHONDONTWRITEBYTECODE=1 python -m unittest tests.test_kb_checks -v
PYTHONDONTWRITEBYTECODE=1 python scripts/check_kb.py
```

An unavailable import must be reported as unrun, never pass. Do not use
`unittest discover` for the product-only command because it also selects checker
tests. Do not use `compileall` as behavioral proof.

## External Ground Truth

No external source determines repository coverage. Exact assertions, selected
workflow commands, inspected artifacts, and named execution results are the
ground truth for claims on this page.
