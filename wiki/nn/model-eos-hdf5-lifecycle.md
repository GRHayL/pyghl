# Neural-Network Model and EOS HDF5 Lifecycle

This page owns standalone model schema, EOS identity, model matching/install, EOS embed/remove, and parent-pinned GRHayL loader seam. Python authority is `src/pyghl/_nn_hdf5.py` and `src/pyghl/_nn_train.py`; upstream claims use parent-pinned Git objects, never current advanced checkout.

## Standalone Schema

`NN_HDF5_FORMAT` is `grhayl_nn_c2p_hdf5`; writer stores root attributes `format` and `format_version=2`. Required groups and project datasets are:

| Group | Contents |
| --- | --- |
| `dims` | `in_dim`, `hidden_dim`, `n_hidden`, `out_dim` |
| `meta` | `q_idx`, `s_idx`, `y_eps`, `dx_eps` |
| `scaling` | `x_eps`, `x_kind`, `x_lo`, `x_hi`, `x_invrng`, `out_kind`, `out_lo`, `out_hi`, `out_invrng` |
| `layers` | `W_in`, `b_in`, `W_hid`, `b_hid`, `W_out`, `b_out` |
| optional `audit` | Writer stores audit mapping as group attributes; reader also accepts datasets |
| optional `source_eos` | EOS identity/provenance mapping as datasets |

`read_nn_hdf5_payload()` checks root `format`, then reads the four required groups by name. It does not check root `format_version`; missing groups/datasets surface from h5py lookups. `write_nn_hdf5()` creates parent directories and opens output with `w`, so it truncates an existing model and has no temporary-file rollback.

## EOS Identity

`build_eos_metadata()` records:

- `canonical_md5`: project-defined traversal digest from root attributes plus sorted datasets, each dataset name/dtype/shape/attributes/content;
- `file_md5`: raw file-byte digest;
- `hash_kind="md5-hdf5-content-v1"`, filename, resolved path, and size.

Canonical traversal excludes every dataset whose top-level path is
`grhayl_nn_c2p`. It does not hash group attributes except root attributes. This
design lets model embedding/removal leave project canonical identity unchanged
while raw file identity changes. The implementation reads each included dataset
in full with `ds[()]` before hashing, so peak memory is proportional to at least
the largest dataset. These MD5 values are matching/provenance identifiers, not
authentication or KB freshness metadata.

Installed matching recomputes target EOS canonical MD5 and reads installed
standalone models in sorted path order. An uncaught read or schema exception
from an earlier file aborts the search instead of skipping that file; otherwise
the first exact `source_eos/canonical_md5` match is returned. Matching does not
compare raw MD5, filename, size, model architecture, or EOS physics.

## Install and Installed Models

`install_nn_model()` requires a source file, parses standalone payload, requires a 32-lowercase-hex source EOS canonical MD5, creates `src/pyghl/nn_c2p/models` relative to installed module, and copies to `<canonical_md5>.h5` with `shutil.copy2`. Existing destination requires `overwrite=True`. Copy is not staged/atomic, rollback is absent, and installed package location may be unwritable.

Four repository files are `source-present`:

- `src/pyghl/nn_c2p/models/20978d67cea9e1803e1c227c1d685440.h5`
- `src/pyghl/nn_c2p/models/45a0d8e40c702bb21be372af9f037821.h5`
- `src/pyghl/nn_c2p/models/b5e790daadfe5341b55b6296bd9d5f4d.h5`
- `src/pyghl/nn_c2p/models/d1cd1380a54cd9d35ee02aee537b59aa.h5`

`pyproject.toml` selects `nn_c2p/models/*.h5` as package data, so all four are
also `package-selected`. No wheel/sdist was inspected, so none is `packaged`. No
per-file schema, loader, numerical, or EOS-compatibility evidence is established
here; those properties and validity remain unproved for every file.

## Append, Overwrite, and Remove

`append_nn_to_eos_file(eos, model)` reads standalone payload and current EOS metadata first. With default `require_eos_match=True`, source EOS metadata must exist, hash kind must equal project kind, and canonical MD5 must match. `--force` in append CLI disables only this matching gate.

An existing, metadata-readable `grhayl_nn_c2p` requires `overwrite=True`.
`append_nn_to_eos_file()` reads required embedded metadata before mutation, so a
malformed or partial group can raise before overwrite can delete it. After that
gate, mutation opens EOS with HDF5 append mode, deletes any existing group, and
creates the replacement. It writes embedded format/version, standalone raw
MD5/filename, EOS hash kind, model groups, optional audit datasets, and
provenance including current EOS metadata, UTC embed timestamp, and model-source
fields. Mutation is in-place and non-transactional: write failure after
deletion/creation can leave missing or partial embedded state.

`append_matching_installed_nn_to_eos_file()` finds exact canonical match then calls append with matching required. Absence reports target and known canonical hashes.

`remove_nn_from_eos_file()` requires a metadata-readable embedded group, captures
before metadata, deletes the group in HDF5 append mode, recomputes metadata, and
reports raw/canonical before/after. Its metadata pre-read can block deletion of
a malformed or partial group. It has no backup, confirmation, or rollback. CLI
first treats absence as a no-op, but direct function raises.

## Parent-Pinned Loader Seam

Parent gitlink object contains:

- `GRHayL/Con2Prim/Tabulated/neural_network_guess/c2p_nn_load_from_eos_hdf5.c`
- `GRHayL/Con2Prim/Tabulated/neural_network_guess/c2p_nn_validate_model.c`
- `GRHayL/Con2Prim/Tabulated/neural_network_guess/ghl_c2p_nn.h`

Pinned loader opens standalone file at root or EOS file under
`grhayl_nn_c2p`, reads exact scalar/array ranks and dimension-derived shapes,
limits dimensions to at most 8 during load, and replaces `eos->c2p_nn` only
after validation succeeds. Validator requires `in_dim=4`, positive dimensions,
in-range q/s indices, finite positive epsilons, finite ordered input scaling
with positive inverse ranges, recognized kinds, and finite weights/biases.
Output index zero must use bounded kind, but its `out_lo`, `out_hi`, and
`out_invrng` values are not numerically validated; later outputs must use
linear/log-linear kinds with finite ordered scaling and positive inverse ranges.
HDF5-disabled builds return a disabled-HDF5 error.

Pinned loader does not read Python standalone root `format`/`format_version`, embedded format/version datasets, audit, source-EOS metadata, or MD5 provenance. For one output it can synthesize bounded-output scaling when `scaling/out_kind` is absent. Therefore Python schema inspection and hash matching do not prove GRHayL load, and GRHayL structural load does not prove EOS identity or numerical behavior.

## Mutation and Artifact Matrix

| Artifact | Producer | Consumer | Location | Tracked/user-owned | Mutation/overwrite | Guard/cleanup | Strongest proof |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Standalone NN HDF5 | trainer/converter | Python lifecycle, GRHayL loader | Caller path | User-owned/generated; four tracked source files | `w` truncates | Python format/required mappings; no temp rollback | Writer `implemented`; tracked files only `source-present`/`package-selected` |
| EOS table with embedded NN | append/train | GRHayL embedded loader, metadata CLI | User EOS path | User-owned | In-place create/delete/replace group | Canonical match by default; overwrite flag; no backup/rollback | `implemented` only |
| Installed-model copy | install/training | matcher/list/append | Module `nn_c2p/models` | Package-managed or local install mutation | Existing destination needs overwrite | Source EOS MD5 syntax; `copy2`; no rollback | `implemented` only |
| Canonical/raw/provenance fields | lifecycle code | matcher/diagnostics | HDF5 datasets/CLI output | Product metadata | Recomputed on operations | Project traversal definition | `implemented`, non-security identity |

## Evidence and Gaps

Python schema/mutations and exact parent-pinned loader/validator were source-traced with `git ls-tree`, `cat-file`, `grep`, and `show`. Current advanced submodule checkout was not used as authority. No HDF5 model was opened, copied, appended, removed, or loaded; no built artifact or numerical test exists in this evidence packet.

## Change Impact

Dataset name/type/shape, kind, dimension, or epsilon changes require Python reader/writer, both export paths, converter, metadata commands, and parent-pinned loader/validator review. Identity changes require matching, installed filenames, append/remove invariance, and migration plan. Any mutation hardening must cover partial writes and user EOS backup/atomicity.

## External Ground Truth

- [h5py File objects](https://docs.h5py.org/en/stable/high/file.html) defines `r`, `w`, and `a` modes used for read, truncate/create, and read/write/create behavior.
- [Python `hashlib`](https://docs.python.org/3/library/hashlib.html) documents MD5/SHA-1 constructors, hexadecimal digests, collision weaknesses, and possible policy restrictions. Project hashes are non-security identifiers.
