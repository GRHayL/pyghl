# CLI Command Workflows

This page owns ordered CLI decisions, mutations, guards, outputs, cleanup, and exit boundaries. Authority is `src/pyghl/cli.py` plus delegated modules under `src/pyghl/nn_c2p/`; download trust details remain in [EOS catalog and download](eos-catalog-and-download.md).

## Training Workflow

Installed `pyghl train` first requires compiled bindings, then delegated flow is:

1. Parse optional EOS, optional dataset, output/training parameters, append/register choices, overwrite choices, force retrain, generation size/mode/scan settings.
2. If EOS omitted, fetch catalog, run interactive selection, and download into current directory. `KeyboardInterrupt` prints cancellation and returns 130; selection `OSError`, `RuntimeError`, or `ValueError` returns 1. Explicit EOS bypasses network selector.
3. Unless `--force_retrain`, inspect EOS embedded metadata.
   - Existing embedded group plus no `--overwrite_eos`: print skip and return 0.
   - No embedded group plus canonical-matching installed model: append it when `--append_eos yes`, otherwise skip; return 0. This path does not train or write requested final model outputs.
4. If no dataset, reserve unique temp name with `mkstemp`, close descriptor, unlink empty file, generate a `train` dataset there using requested `n_pts`, target mode, and scan points.
5. Call `ghl.nn.train_on_dataset()`. That core repeats embedded/installed short circuits, reads dataset, trains, then writes bundle, standalone HDF5, header, optional EOS embed, and optional installed-model copy in order.
6. `finally` deletes only auto-generated temp dataset. Explicit dataset, downloaded EOS, final outputs, logs, and checkpoints remain.

`--force_retrain` bypasses early skip but does not itself authorize embedded replacement. Existing embedded EOS plus append requires `--overwrite_eos`; installed destination replacement separately requires `--overwrite_installed_model`. `--append_eos no` and `--register_installed_model no` disable those mutations.

No cross-artifact transaction exists. Later failure can leave downloaded EOS, logs/checkpoints, bundle/HDF5/header, embedded mutation, or installed copy already changed. Training core catches installed-copy errors only, prints warning, and can still return success after other outputs.

## Append Workflow

`pyghl append EOS [MODEL]` reads embedded metadata first. Existing group without `--overwrite` exits via `SystemExit` before mutation.

- No model argument: find installed model by canonical EOS MD5 and append with match required. Absence exits with training suggestions.
- Explicit model: append it; source-EOS canonical match is required unless `--force`.
- `--force` disables EOS checksum verification only. It does not bypass format reads or existing-group overwrite guard.
- `--overwrite` deletes and recreates embedded group in place; operation is non-atomic.

Source dispatch also accepts `pyghl append-eos`, but it is not listed in top-level subparser/help metadata.

## Inspect, List, and Remove

- `check-eos EOS` opens file read-only through metadata helper, reports group presence, format/shape, provenance, and hashes when present; absence returns 0.
- `list-models` scans sorted `*.h5` files in installed model directory and reads every payload. Empty set returns 0. A malformed/unreadable model is not skipped by source.
- `remove-eos-nn EOS` first reads metadata. Absence prints no-op and returns 0. Presence deletes `grhayl_nn_c2p` in place; `--verbose` prints raw/canonical before/after. No confirmation, backup, force, or rollback exists.

These commands do not call `require_bindings()` in top-level dispatcher, but they still require importable `pyghl` Python package and HDF5 dependencies. Source registration does not prove installed execution.

## Module Tools and Examples

| Tool | Inputs | Effects and guards |
| --- | --- | --- |
| `nn_c2p_generate_dataset` | EOS, train/test, size/mode/scan | Native/EOS computation; truncates binary output; validates basic options/nonfinite rows |
| `nn_c2p_test` | EOS, dataset, optional model/output/range | Loads embedded model or explicit fallback, runs native recovery, truncates text output |
| `header_to_hdf5` | generated header, output, optional EOS | Parses trusted project-style header, truncates HDF5 output; no parity proof |
| `_nn_train` | EOS and optional dataset | Older/direct orchestration with its own parser; may generate/train/embed/register |
| `examples/nn_c2p_*.py` | delegated arguments | Thin wrappers around module tools; operational, not proof |
| `scripts/train_all.sh` | POSIX shell glob `*.h5`; if nothing matches, the literal `*.h5` is still passed to one training invocation | Repeated expensive training; writes `/tmp`; overwrites installed matches; no `set -e` or status aggregation means intermediate failures do not stop later iterations and may be hidden by the final command status |

## Artifact and Mutation Matrix

| Artifact | Producer | Consumer | Location | Tracked/user-owned | Mutation/overwrite | Guard/cleanup | Strongest proof |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Downloaded EOS | remote train selection | generator/training/user | Current destination path; reused symlink may resolve elsewhere | User-owned/cache | `is_file()` true reuses path; dangling symlink is replaced by final publication | Trust/size/decompression gates; no cache/symlink validation or second destination check; partials cleaned | `implemented`; live availability unproved |
| Auto dataset | train CLI | trainer | OS temp directory | Temporary | Unique reserved name then generated | `finally` unlink | `implemented` |
| Explicit dataset | User/generator | trainer | User path | User-owned | CLI does not modify | Not cleaned | `implemented` |
| Log/checkpoint/final outputs | trainer/exporters | resume/inference/C/user | Supplied/default paths | User-owned/generated | May truncate/overwrite | Per-stage checks; no transaction | `implemented` |
| EOS embedded group | append/train/remove | GRHayL/metadata | User EOS HDF5 | User-owned mutation | Create/overwrite/delete in place | Match and overwrite gates; no rollback | `implemented` |
| Installed model | training/install | matcher/list/append | Package module model directory | Package/local-install mutation | Overwrite flag | Canonical source MD5 syntax; no rollback | `implemented` |
| Evaluation `.asc` | evaluator | User | Default/explicit path | User-owned/generated | Text `w` truncates | No temp/cleanup | `implemented` |

## Exit and Failure Boundaries

Top-level no-arg status is 2. Help/version as the first token returns 0 without
parsing or rejecting trailing tokens. Unknown command/parser misuse returns 2;
train binding failure returns 1. Remote cancellation is explicitly 130 and
expected selection failures 1. Successful skip/no-op paths return 0. Delegated
argparse exits, `SystemExit` messages, dependency errors, HDF5 errors, native
`GRHayLError`, filesystem failures, and training exceptions otherwise propagate;
there is no global exception-to-status policy.

## Evidence and Gaps

All ordered branches above are source-traced. `tests/test_nn_c2p_train_cli.py` mocks EOS selection to prove explicit/omitted resolution and cancellation 130. No test executes installed dispatch, early model append, training, output order, rollback, append/remove, evaluator, converter, or batch script. No command with network, generation, training, or mutation ran for KB work.

## Change Impact

Flow changes require early-return side effects, distinct force/overwrite meanings, temp cleanup under every exception, and artifact order review. Parser defaults must stay synchronized with direct core behavior and documentation. Catalog trust changes belong in [EOS catalog and download](eos-catalog-and-download.md); HDF5 mutation semantics belong in [model lifecycle](../nn/model-eos-hdf5-lifecycle.md).

## External Ground Truth

- [Python `tempfile.mkstemp`](https://docs.python.org/3/library/tempfile.html#tempfile.mkstemp) defines secure temporary creation and caller-owned descriptor/file cleanup used by auto-dataset orchestration.
- [Python `argparse`](https://docs.python.org/3/library/argparse.html) defines delegated parse and `SystemExit` behavior.
