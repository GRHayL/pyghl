# Command-Line Knowledge Hub

This hub routes installed command, module-tool, example, training orchestration, and remote EOS questions. Source registration and dispatch are evidence only for declared/implemented interfaces; they do not prove an installed script, usable bindings, live network service, or successful execution.

## Installed Surface

`pyproject.toml` declares one console script, `pyghl = pyghl.cli:main` (`selected`). `src/pyghl/cli.py` implements raw first-argument dispatch (`registered` in source):

| Command | Owner | Primary mutation/effect | Binding prerequisite at dispatcher |
| --- | --- | --- | --- |
| `pyghl train` | [Command workflows](command-workflows.md) | May download EOS, generate/train, write artifacts, embed/register model | Explicit `require_bindings()` before delegated import |
| `pyghl append` | [Command workflows](command-workflows.md) | Mutates EOS HDF5 in place | No explicit dispatcher gate; Python/HDF5 stack required |
| `pyghl check-eos` | [Command workflows](command-workflows.md) | Read-only EOS metadata | No explicit dispatcher gate |
| `pyghl list-models` | [Command workflows](command-workflows.md) | Reads installed model files | No explicit dispatcher gate |
| `pyghl remove-eos-nn` | [Command workflows](command-workflows.md) | Deletes embedded EOS group | No explicit dispatcher gate |
| `pyghl --version` | This page | Reads installed distribution metadata; prints `unknown` if absent | None |

`append-eos` is a source-visible alias in dispatch even though only `append` is added to top-level subparser metadata. Dispatcher does not call top-level `parse_args()` for commands; delegated parsers validate remaining arguments, so alias reaches append implementation.

No args prints help and returns 2. When `-h/--help` is the first token, dispatch
prints help and returns 0 without parsing or rejecting trailing tokens;
`-v/--version` behaves the same way for version output. Unknown first argument
calls `parser.error` and exits 2. Training binding failure calls
`parser.exit(1, ...)`. Delegated parsers and uncaught operational exceptions
determine other exits.

## Other Runnable Source Surfaces

These are module/example source entry points, not additional installed scripts:

- `python -m pyghl.nn_c2p.nn_c2p_generate_dataset`
- `python -m pyghl.nn_c2p.nn_c2p_train`
- `python -m pyghl.nn_c2p.nn_c2p_test`
- `python -m pyghl.nn_c2p.header_to_hdf5`
- `python -m pyghl.nn_c2p.append_eos_file`, `check_eos`, `list_installed_models`, `remove_eos_nn`
- `python -m pyghl._nn_train` uses a distinct direct-training parser/flow
- `examples/nn_c2p_generate_dataset.py`, `nn_c2p_train.py`, and `nn_c2p_test.py` are thin operational wrappers, not tests
- `scripts/train_all.sh` passes each current-directory `*.h5` match to training;
  with no match, POSIX shell passes one literal `*.h5`. It writes final/checkpoint
  artifacts under `/tmp`, registers with overwrite, and does not append to EOS.
  No `set -e` or status aggregation stops later iterations, so final success can
  mask an earlier failure.

All generation, evaluation, training, conversion, append, remove, and batch surfaces may be expensive or destructive. Read their owner page before running.

## Routes

| Query/task/path | Canonical owner | Primary authority | Proof/workflow route |
| --- | --- | --- | --- |
| Parser/dispatch, train short circuits, outputs, mutation flags, cleanup | [Command workflows](command-workflows.md) | `src/pyghl/cli.py`, `src/pyghl/nn_c2p/*.py` | Ordered flow and artifact matrix |
| Catalog crawl, terminal selector, trust checks, download/decompression | [EOS catalog and download](eos-catalog-and-download.md) | `src/pyghl/nn_c2p/eos_catalog.py` | Mocked/temp-file tests and gaps |
| Dataset generation details | [NN dataset and generation](../nn/dataset-and-generation.md) | Generator/readers | Producer/reader contract |
| Training/checkpoint internals | [NN training and checkpoints](../nn/training-and-checkpoints.md) | `_nn_train.py`, `_nn_common.py` | Ordered training pipeline |
| Model mutation/identity | [NN HDF5 lifecycle](../nn/model-eos-hdf5-lifecycle.md) | `_nn_hdf5.py` | Mutation and loader seams |

## Evidence and Gaps

Console-script metadata is `selected`, not `packaged` or `executed`. Dispatch and delegated parser paths are `implemented`. `tests/test_nn_c2p_train_cli.py` directly proves only optional EOS parsing/resolution and cancellation status through mocks; catalog tests are bounded on their owner page. No root test invokes installed `pyghl`, validates all exit paths, or executes model mutations.

## Change Impact

Top-level command spelling needs `pyproject.toml`, dispatcher, delegated parser, README, tests, and shell-script review. New mutation flags need lifecycle guards and artifact cleanup. New remote behavior belongs in catalog/download owner, not duplicated in command flow.

## External Ground Truth

- [Python `argparse`](https://docs.python.org/3/library/argparse.html) defines parser help/error/exit machinery used by project dispatchers.
