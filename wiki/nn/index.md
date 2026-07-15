# Neural-Network Knowledge Hub

This hub routes neural-network dataset, training, inference, export, and HDF5 lifecycle questions to one canonical owner. Repository source is authority; this page claims source-visible behavior only, not numerical validity, Python/C parity, package inclusion, or runtime success.

## Read First

| Query/task/path | Canonical owner | Primary authority | Proof/workflow route |
| --- | --- | --- | --- |
| Dataset binary layout, generation modes, columns, scale, cleanup | [Dataset and generation](dataset-and-generation.md) | `src/pyghl/_nn_dataset.py`, `src/pyghl/nn.py`, `src/pyghl/nn_c2p/nn_c2p_generate_dataset.py` | Producer/reader trace and artifact table |
| Transforms, split, optimization, checkpoints, logs | [Training and checkpoints](training-and-checkpoints.md) | `src/pyghl/_nn_common.py`, `src/pyghl/_nn_train.py` | Pipeline trace and coverage gaps |
| PyTorch bundles, prediction, C headers, header conversion | [Inference and export](inference-and-export.md) | `src/pyghl/_nn_infer.py`, `src/pyghl/_nn_train.py`, `src/pyghl/nn_c2p/header_to_hdf5.py` | Representation matrix and pinned C seam |
| Standalone/embedded HDF5, EOS identity, installed models | [Model and EOS HDF5 lifecycle](model-eos-hdf5-lifecycle.md) | `src/pyghl/_nn_hdf5.py`, parent-pinned GRHayL loader | Mutation matrix and evidence limits |
| Installed command decisions and remote EOS selection | [CLI workflows](../cli/command-workflows.md) | `src/pyghl/cli.py`, `src/pyghl/nn_c2p/` | Command flow and exits |
| Catalog/download trust boundary | [EOS catalog and download](../cli/eos-catalog-and-download.md) | `src/pyghl/nn_c2p/eos_catalog.py` | Mocked/temp-file tests |

`src/pyghl/nn.py` is the public Python facade. Its dataset dataclasses and readers do not require Torch, while `_load_training_api()` imports optional training modules on demand. `guess()` and `guess_x()` require compiled bindings and a loaded `TabulatedEOS`; training helpers have a different dependency and artifact boundary.

## End-to-End Artifact Route

1. Generator writes a binary dataset; readers select `q,r,s,t` and one target.
2. Training creates model state plus transform/scaling metadata, logs, and optional checkpoints.
3. Export creates a PyTorch inference bundle, standalone HDF5, and optional C header.
4. Standalone HDF5 may be copied into the installed-model directory or embedded into an EOS file.
5. Parent-pinned GRHayL reads either standalone datasets or the EOS `grhayl_nn_c2p` prefix for C inference.

Each transition has an independent proof state. Source-visible production of one representation does not prove another representation is equivalent or loadable.

## Evidence Summary

- `implemented`: producer, readers, training, export, lifecycle, and CLI paths exist in source.
- `exported`: `src/pyghl/nn.py` exposes facade functions; this does not prove import or execution.
- `coverage-gap`: no root test directly proves dataset generation, transforms/training, bundle inference, Python/C parity, HDF5 lifecycle, or tracked-model compatibility.
- Tracked models are handled only as stated in [Model and EOS HDF5 lifecycle](model-eos-hdf5-lifecycle.md).

## Change Impact

Changes under `_nn_common.py`, `_nn_dataset.py`, `_nn_train.py`, `_nn_infer.py`, `_nn_hdf5.py`, or `nn_c2p/` require review of every downstream representation they feed. Schema or numerical-path edits also require parent-pinned loader review; CLI edits require parser, mutation, cancellation, and cleanup review.

## External Ground Truth

- [PyTorch reproducibility notes](https://docs.pytorch.org/docs/stable/notes/randomness.html) bound what seeding and deterministic settings can prove.
- [PyTorch `torch.load`](https://docs.pytorch.org/docs/stable/generated/torch.load.html) defines `weights_only` and `map_location` behavior used by checkpoint and bundle readers.
- [h5py File objects](https://docs.h5py.org/en/stable/high/file.html) define HDF5 open modes used by lifecycle code.
