# Neural-Network Training and Checkpoints

This page owns data preparation, model fitting, checkpoint/resume, logging, and final training artifacts. `src/pyghl/_nn_train.py` and `src/pyghl/_nn_common.py` are authority; no training run or reproducibility guarantee is evidence for this page.

## Entry Points and Preconditions

`train_regressor(data_np, ...)` is the direct in-memory core. `train_on_dataset(path, ...)` reads the project binary dataset, applies EOS/model short circuits, invokes the core, and exports artifacts. `src/pyghl/nn.py` lazily exports both. Installed `pyghl train` adds EOS selection, optional dataset generation, and cleanup; keep those orchestration decisions separate from core fitting.

Torch and NumPy must import. Dataset must be 2-D, have more columns than `n_out`, retain at least two rows after filtering, provide one target, and include configured `q_idx`/`s_idx`. Accepted target modes are `x_correction` and `x_best_correction`; both are treated as one bounded-x target.

## Ordered Core Pipeline

1. Set Torch default dtype to float32; seed Torch and Python `random`. Create seeded generators for split and epoch permutations.
2. If `deterministic=True`, attempt `torch.use_deterministic_algorithms(True)`, set cuDNN deterministic, and disable cuDNN benchmark. Exceptions are swallowed. This is a best-effort configuration, not a guarantee.
3. Pick CUDA, then available/built MPS, otherwise CPU.
4. Convert input to float32 tensor. Drop nonfinite rows by default or fail; require at least two remaining rows.
5. Reject or drop rows whose Palenzuela width `1+q` is nonfinite or at most `width_tiny`; clamp target x inside epsilon-shrunk bounds derived from `q,s`.
6. Compute `n_val=max(1,min(int(val_frac*N),N-1))`; seeded `randperm` assigns first indices to validation and rest to training. SHA-1 over permutation bytes is truncated to 16 hex characters as `perm_sha1_16`.
7. Convert target x to bounded `[y_eps,1-y_eps]` coordinates. Output scaling for this target is identity over `[0,1]`.
8. Fit feature kinds on training inputs only. A feature becomes log10 only when it has finite positive support, acceptable negative fraction, enough positive samples, a safe 1% quantile, and 99%/1% ratio above threshold; `force_kind` can override by index.
9. Apply identity/log10 transform to train and validation inputs. Fit per-feature robust quantile min/max on transformed training data; clip and scale both splits. Standardization statistics are also recorded but current prediction uses robust min/max fields.
10. Move working tensors to selected device. Construct `TinyMLP_Logit`: input linear plus HardTanh, `n_hidden-1` hidden linear/HardTanh stages, then linear logits. CUDA attempts `torch.compile`; failure falls back.
11. Optimize with AdamW (fused attempt on CUDA), optional gradient clipping, and `ReduceLROnPlateau`. Loss is selected relative-error loss on decoded x. Validation tracks relative metrics and y-coordinate MSE.
12. Replace best state only when relative-RMSE improvement exceeds `min_delta`; stop after `patience` consecutive non-improving epochs. Restore best state before return when one exists.

Returned model/stats are float32 and normally moved to CPU. Device is returned separately.

## Determinism and Provenance Boundary

`perm_sha1_16` identifies split permutation bytes for product run provenance. It
is written to console, log metadata, and checkpoints; it is not a security
digest and never represents KB freshness. A fixed seed and requested
deterministic algorithms constrain a run, but PyTorch does not guarantee
complete reproducibility across releases, platforms, or CPU/GPU. Code also
catches deterministic-setting failures and may use compiled/fused/device-specific
paths. `train_regressor()` sets the process-global Torch default dtype and
Torch/Python RNG state; when deterministic mode is requested and assignments
succeed, it also changes deterministic-algorithm and cuDNN flags.
`train_on_dataset()` sets process-global float32 matmul precision. Source does
not restore prior values.

## Checkpoint and Resume Flow

Positive `checkpoint_every` creates `checkpoint_dir`. At matching epochs, code writes `<prefix>_epNNNNN.pt` containing current/best model states, optimizer and scheduler states, epoch/best metrics, learning rate, `perm_sha1_16`, RNG snapshots, per-epoch generator state, backend flags, transform/scaling stats, and loss configuration. Optional same-epoch C-header export follows; checkpoint save succeeds before header export is attempted.

Resume accepts a file or directory. Directory resolution chooses maximum parsed `_epNNNNN.pt` epoch then modification time. `torch.load(..., map_location="cpu", weights_only=True)` must exist in installed Torch; returned object must be a dict with `model_state_dict`. Model keys normalize `_orig_mod.` or `module.` prefixes. Optimizer/scheduler, best state, selected RNG states, and epoch are restored when present; incomplete optional state prints notes or continues.

Important boundary: split and transform/scaling statistics are freshly recomputed before checkpoint load. Stored `ft_stats`, `x_stats`, and `y_stats` are not returned by `_load_training_checkpoint()` and do not replace current values. Resume therefore depends on caller supplying the same intended data and configuration; source contains no dataset/config identity gate.

## Logs and Final Orchestration

New training opens `log_path` with `w`; resume uses `a`. Header metadata includes seed, `perm_sha1_16`, indices, target mode, bounds/filter settings, clip fractions, and loss configuration. Epoch rows include validation/train metrics, best epoch, clip fraction, and learning rate. No overwrite prompt or atomic log write exists.

`train_on_dataset()` may short-circuit before reading/training when EOS already embeds a model or when a canonical-EOS-matching installed model exists. Otherwise it sets float32 matmul precision, trains, then conditionally writes bundle, standalone HDF5, C header, embedded EOS group, and installed-model copy in that order. Failures do not roll back earlier artifacts. Installed-model registration errors are caught and printed as warnings after other outputs.

| Artifact | Producer | Consumer | Location | Tracked/user-owned | Mutation/overwrite | Guard/cleanup | Strongest proof |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Training log | `train_regressor()` | User | `log_path` | User-owned/generated | New run truncates; resume appends | No atomic write/cleanup | `implemented` |
| Checkpoint `.pt` | `train_regressor()` | resume loader | `checkpoint_dir` | User-owned/generated | Same epoch path overwritten by `torch.save` | `weights_only=True` read; no temp/rollback | `implemented` |
| Checkpoint `.h` | `export_to_c_header()` | C build or converter | checkpoint directory | User-owned/generated | Existing path truncated | Shape/finite checks before write; no rollback | `implemented` |
| Best in-memory state | training loop | final return/export | Process memory | Ephemeral | Replaced on sufficient improvement | Restored after loop | `implemented` |
| Final bundle/HDF5/header | `train_on_dataset()` | Python/C/lifecycle consumers | Caller/default paths | User-owned/generated | Existing files may be overwritten | Per-export validation only; no cross-artifact transaction | `implemented` |

## Evidence and Gaps

Training pipeline and artifact order are source-traced. Root tests do not exercise transforms, devices, optimization, early stop, checkpoints, resume compatibility, logs, or output rollback. No training was run; convergence, speed, reproducibility, scientific accuracy, and artifact compatibility remain unproved.

## Change Impact

Transform, bounds, target, architecture, or stored-stat changes require [Inference and export](inference-and-export.md), [HDF5 lifecycle](model-eos-hdf5-lifecycle.md), and parent-pinned loader/validator review. Resume schema changes require old/new checkpoint fixtures. Output order changes require [CLI workflows](../cli/command-workflows.md) mutation and cleanup review.

## External Ground Truth

- [PyTorch reproducibility](https://docs.pytorch.org/docs/stable/notes/randomness.html) states that complete cross-release/platform reproducibility is not guaranteed and documents seeds/deterministic settings.
- [PyTorch `torch.load`](https://docs.pytorch.org/docs/stable/generated/torch.load.html) defines `weights_only` restrictions and `map_location` remapping used by resume.
- [Python `hashlib`](https://docs.python.org/3/library/hashlib.html) documents SHA-1 and its collision weakness; project uses its truncated output only as run provenance.
