# Generated, Mutable, and External Artifact Boundaries

This page owns the cross-cutting artifact ledger: who produces and consumes each
file/tree, where it lives, who owns it, how it mutates, and what evidence exists.
Source/configuration rows describe possible production; no artifact is claimed
built, packaged, imported, executed, or published without exact inspection.

## Rules Before Producing Anything

- Read the domain owner and [workflows](workflows.md). Record prerequisites,
  network/cost, destination, overwrite behavior, cleanup, and proof goal.
- Preserve pre-existing tracked, untracked, ignored, external-checkout, and
  nested-repository state. Never delete an output merely because `.gitignore`
  matches it.
- Prefer disposable destinations outside the repository. Delete only artifacts
  proven created by the authorized run; never blanket-clean shared paths.
- `package-selected` is metadata. `packaged` requires contents of an exact named
  archive. `artifact-inspected` proves only examined fields, not loader or
  numerical compatibility.

## Build and Package Artifacts

| Artifact | Producer | Consumer | Location | Tracked/user-owned | Mutation/overwrite | Guard/cleanup | Strongest proof |
| --- | --- | --- | --- | --- | --- | --- | --- |
| GRHayL generated `Makefile` | GRHayL `configure`, called by `setup.py:_run_make_grhayl` when Makefile/build absent | GRHayL `make grhayl` | Selected `GRHAYL_ROOT/Makefile`, default `extern/GRHayL/Makefile` | Generated inside nested or external user checkout | Configure removes/recreates Makefile; may reconfigure existing source state | Preserve checkout first; only owner-approved GRHayL cleanup | `implemented` in `setup.py` and parent-pinned `configure`; not run |
| GRHayL object/build tree | Parent-pinned or external GRHayL configure/make | `libghl` link; pyghl extension link | `GRHAYL_ROOT/build/**` | Generated nested/external user state | `make` updates objects/libraries; setup always invokes target | Never reset/clean nested/external checkout as incidental cleanup | `implemented`; not compiled here |
| `libghl` versioned/unversioned shared libraries | GRHayL `make grhayl` | Extension linker; non-inplace copy; runtime loader | `GRHAYL_ROOT/build/lib/libghl*.so` or `*.dylib` | Generated nested/external state; copied package artifacts differ | Make/symlink updates build libs; non-inplace `copy2` replaces destination file | Use disposable checkout/build when authorized; inspect exact copy/link | `selected` by `setup.py`; no compile/link claim |
| CPython extension | setuptools `BuildExt` compiling `csrc/pyghl_module.c` | `src/pyghl/__init__.py` import and Python callers | Inplace/package build destination as `pyghl/_pyghl` platform extension | Generated build/environment artifact | Build backend replaces output | Fresh environment/output; uninstall/discard run-owned tree | `package-selected` by `Extension`; no built artifact inspected |
| Copied runtime `libghl` | Non-inplace `BuildExt.run()` | Extension dynamic loader | Beside built `pyghl/_pyghl`; recognized `.so`/`.dylib` names | Generated package-build artifact | `shutil.copy2` replaces same destination | Inspect exact build/wheel; discard disposable output | `package-selected` by copy and package-data rules; not `packaged` |
| macOS rewritten extension/libraries | `install_name_tool` in non-inplace build; delocate in workflows | macOS runtime loader | Package build tree, then repaired wheel | Generated binary artifact | Rewrites install IDs/load commands; repair writes destination wheel | Keep original/repaired artifacts distinct; inspect architecture/dependencies | `selected` by setup/workflow; no repair run observed |
| setuptools build and metadata trees | PEP 517/setuptools/pip | Build backend, installer, packaging inspection | Common `build/`, `*.egg-info/`, temporary frontend dirs | Generated; root `.gitignore` matches `build/`, `*.egg-info/` | Backend recreates/updates | Fresh outside-repo frontend output when possible; remove only run-owned | Build backend `selected`; no build run claimed |
| Source distribution | setuptools backend/frontend | Wheel builder, installer/downstream packager | Usually requested `dist/` or external output, `pyghl-<version>.tar.gz` | Generated distribution; `dist/` ignored | New archive may replace/collide by frontend behavior | Use empty disposable output; inspect members then discard | `package-selected` inputs from `MANIFEST.in`/metadata; no sdist inspected |
| Pre-repair Linux/macOS wheel | cibuildwheel or local wheel frontend before repair | Repair step or local inspector/installer | cibuildwheel transient path supplied as `{wheel}`, or explicitly requested local pre-repair output; not workflow `wheelhouse/` | Generated distribution | Build writes version/tag-named archive | Keep pre-repair and repaired identities distinct; inspect exact archive/tags | Build path `selected`; no pre-repair wheel inspected |
| Repaired Linux/macOS wheel | cibuildwheel default Linux `auditwheel` or configured macOS `delocate-wheel` | Test command, upload artifact, installer | `{dest_dir}`; workflow `wheelhouse/*.whl` | Generated distribution | Repair creates destination artifact before testing | Retain identity between pre/post repair; inspect dependencies and tags | Workflow repair `selected`; no repaired wheel inspected |
| GitHub wheel artifact | `actions/upload-artifact` from each matrix job | Release publish download or human inspection | Hosted artifact `wheels-linux-x86_64` / `wheels-macos-arm64` | Hosted run-owned object | Upload action creates artifact; platform retention/deletion applies | Name exact run/artifact; do not infer from YAML | Configured/`selected`; no hosted artifact observed |
| Downloaded publish `dist/` tree | `actions/download-artifact` with merge | PyPI publishing action | Publish runner `dist/` | Ephemeral hosted run state | Merge collects wheel artifacts | Exact run inspection required; runner cleanup platform-owned | Configured/`selected`; no run observed |
| Published release files | `pypa/gh-action-pypi-publish` after release event | PyPI installers/users | PyPI project/version storage | Externally hosted immutable-by-filename artifacts | Upload creates project files; same filename/version cannot be replaced per maintainer docs | Verify version/tag/artifact set before publish; rollback needs new release/version policy | Publish action `selected`; no `release-published` claim |

`MANIFEST.in` selects `README.md`, `csrc/*.c`, and `src/**/*.py`, `*.typed`,
`*.h5` for source-distribution input. `pyproject.toml` package-data selects
`libghl*.so*`, `libghl*.dylib`, and `nn_c2p/models/*.h5`; `setup.py` selects the
extension and runtime-library copy. None proves archive contents.

## Dataset, Training, and Model Artifacts

| Artifact | Producer | Consumer | Location | Tracked/user-owned | Mutation/overwrite | Guard/cleanup | Strongest proof |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Direct training/test dataset | `generate_dataset()` / module CLI | `_nn_dataset.read_training_dataset`, training, manual C2P test | Explicit output or `nn_training_dataset.bin` / `nn_test_dataset.bin` in current directory | User-owned generated binary | Opened `wb`; existing file is truncated | Choose fresh path; direct generator has no automatic cleanup | Producer/consumer `implemented`; no dataset generated |
| Autogenerated train dataset | `pyghl train` via `tempfile.mkstemp` then generator | Same training invocation | System temp, `grhayl_nn_training_<stem>_*.bin` | Temporary run-owned | Reserved file removed before generator opens output | `finally` removes if present; interruption before `try` boundaries still merits review | Cleanup `implemented`; not executed |
| Training log | `_nn_train.train_regressor` | Human audit/resume comparison | Default `training_log.txt` or `--log_path` | User-owned generated text | Fresh training opens `w`; resume opens `a` | Use fresh path/back up; no automatic deletion | `implemented`; no log produced |
| Checkpoint directory/files | `_nn_train.train_regressor` | Resume loader, human recovery | Default `checkpoints/<prefix>_epNNNNN.pt` | User-owned generated Torch artifacts | Directory created; same epoch path overwritten by `torch.save` | Fresh directory; caller removes only owned outputs | `implemented`; no checkpoint produced |
| Checkpoint C headers | Training checkpoint export | C consumer/manual inspection | `checkpoints/<prefix>_epNNNNN.h` | User-owned generated source | Header writer opens `w` | Same guard as checkpoints; persistent | `implemented`; no header generated |
| Final inference bundle | `_nn_infer.save_inference_bundle` from `train_on_dataset` | `_nn_infer.load_inference_bundle`, inference callers | Default `tiny_mlp_inference.pt` or `--bundle_output` | User-owned generated Torch artifact | `torch.save` writes target; no collision guard | Fresh path/back up; caller cleanup | `implemented`; no bundle inspected |
| Final standalone NN HDF5 | `_nn_train.export_to_hdf5` / `write_nn_hdf5`; header converter alternative | HDF5 lifecycle, installed cache, EOS append, GRHayL loader seam | Default `tiny_mlp_model.h5` or explicit output | User-owned generated HDF5 | h5py opens `w`, replacing file contents | Fresh path/back up; validate exact fields read-only before mutation | `implemented`; no generated file inspected |
| Final C header | `_nn_train.export_to_c_header` | C integration or header-to-HDF5 converter | Default `tiny_mlp_weights.h` or explicit output | User-owned generated source | Opens `w` | Fresh path/back up; no automatic cleanup | `implemented`; no header inspected |
| Installed model cache entry | `_nn_hdf5.install_nn_model` during training | Model listing/matching/append | `src/pyghl/nn_c2p/models/<canonical-md5>.h5` in editable tree, analogous installed package path | Package-tree mutation; tracked models may already occupy names | Refuses existing unless overwrite flag; `copy2` then replaces | Never use tracked source as scratch; authorize install/overwrite; no auto cleanup | `implemented`; four tracked files are `source-present` and `package-selected` only |
| Four tracked model HDF5 files | Repository contents; selected by package metadata | Installed model discovery/matching | `src/pyghl/nn_c2p/models/*.h5` | Tracked product artifacts | KB work must not mutate | Per-file read-only inspection only; Git preserves history, not runtime validity | `source-present`, `package-selected`; no wheel or compatibility proof |
| Embedded `grhayl_nn_c2p` group | `_nn_hdf5.append_nn_to_eos_file` | Python metadata, parent-pinned GRHayL loader | Inside user EOS HDF5 | User-owned in-place mutation | Existing group rejected unless overwrite; overwrite deletes then recreates | Back up EOS; require model/EOS match unless explicit force; no transaction/automatic rollback | Mutation `implemented`; no EOS mutated |
| Removed embedded group | `_nn_hdf5.remove_nn_from_eos_file` | User EOS without embedded model | Inside user EOS HDF5 | User-owned in-place mutation | Deletes group with no backup | CLI no-ops if absent; back up before removal | Mutation `implemented`; not executed |
| Batch training outputs | `scripts/train_all.sh` | Humans/installed model cache | `/tmp/<stem>_nn.h5`, `.pt`, `.h`, `_training.log`, `_checkpoints/` | User/run-owned; installed model copy touches package tree | Script passes installed-model overwrite; output writers may replace | Shell supplies no cleanup; isolate inputs and clear only run-owned `/tmp` paths | Script `source-present`; not executed |

See [dataset/generation](nn/dataset-and-generation.md), [training/checkpoints](nn/training-and-checkpoints.md),
[inference/export](nn/inference-and-export.md), and [HDF5 lifecycle](nn/model-eos-hdf5-lifecycle.md)
for owner contracts.

## Download, EOS, and Manual-Tool Artifacts

| Artifact | Producer | Consumer | Location | Tracked/user-owned | Mutation/overwrite | Guard/cleanup | Strongest proof |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Compressed EOS partial | Catalog downloader | bzip2/tar+bzip2 decompressor | Destination directory `<remote filename>.part` | Temporary run-owned, possibly large | Stale same-name partial unlinked before download; opened `wb` | `finally` unlinks; preserve unrelated paths | Source cleanup and temp tests `implemented`/asserted; no live download |
| Decompressed EOS partial | Catalog downloader | Atomic final replacement | Destination directory `<EOS filename>.part` | Temporary run-owned, possibly very large | Stale same-name partial unlinked; opened `wb` | `finally` unlinks | Source cleanup and temp tests; no live download |
| Downloaded/cached EOS HDF5 | Catalog downloader | Bindings, generator, trainer, user | Current/destination path; reused symlink may resolve elsewhere | User-owned external data | `is_file()` true reuses path, including valid file symlinks; other `exists()` true raises; dangling symlink passes both gates and final publication replaces the link itself | No final cleanup, cache-content/symlink-target validation, lock, or second destination check; validate/backup before later mutation | Mock/temp bytes asserted; no live artifact or HDF5 compatibility proof |
| Catalog HTML/responses | Remote server and urllib response | Parsers during process | Memory only | External/transient | Not persisted by code | Size/page/host limits; process release | Source/tests; live contents unavailable/unproved |
| Manual C2P ASCII result | `nn_c2p_test.py` | Human comparison/plotting | Explicit output or `nn_test*.asc` in current directory | User-owned generated text | Opened `w` | Fresh path; no automatic cleanup | `implemented`; module source is not test execution |
| EOS plot image | `examples/plot_eps_vs_temp.py` | Human | `--output` path | User-owned generated image | `savefig` writes destination | Fresh path/back up; no cleanup | `implemented`; not executed |
| Rewritten EOS output | `examples/plot_eps_vs_temp.py --output-table` | Human/possible later EOS tools | Explicit resolved path distinct from input; distinct hard links are not detected | User-owned HDF5 unless hard-linked to another owned input | `read_bytes()` materializes the full input before overwriting output, then HDF5 `r+` replaces datasets; a hard-linked output mutates the input inode | Verify different inode, back up, budget memory for full bytes and arrays; no cleanup | `implemented`; no physical validity proof |

## Caches, Bytecode, and Test Output

| Artifact | Producer | Consumer | Location | Tracked/user-owned | Mutation/overwrite | Guard/cleanup | Strongest proof |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Python bytecode/cache | Python import/compile | Python runtime | `__pycache__/`, `*.pyc` | Generated, ignored | Runtime updates | KB commands set `PYTHONDONTWRITEBYTECODE=1`; remove only run-owned cache | Ignore/config only; no run required |
| pytest/cache/coverage output | Test/coverage tools if used | Test tooling/humans | `.pytest_cache/`, `.coverage*`, `htmlcov/`, etc. | Generated, ignored | Tools update | Prescribed KB/product commands use unittest without coverage; preserve pre-existing caches | Ignore patterns only |
| Packaging downloads/caches | pip/build tools | Installer/build frontend | Tool-managed external cache and possible ignored `downloads/` | External/tool/user-owned | Tool policy | Use isolated environment/cache if artifact identity matters; do not clean user cache | Not invoked or inspected |

Root `.gitignore` covers many standard build/test paths, including `build/`,
`dist/`, `*.egg-info/`, `__pycache__/`, `.pytest_cache/`, logs, environments, and
`downloads/`. It does not make outputs disposable and does not visibly cover
default datasets, NN HDF5/bundles/headers/checkpoints, `wheelhouse/`, or arbitrary
EOS/plot outputs. Parent-pinned GRHayL has its own ignore policy; never apply root
cleanup assumptions inside the nested repository.

## External Ground Truth

- [cibuildwheel 2.23.3 repair configuration](https://cibuildwheel.pypa.io/en/v2.23.3/options/#repair-wheel-command)
  defines platform defaults, pre-test repair, and `{wheel}`/`{dest_dir}`
  identities used by the pinned workflow action.
- [PyPA package formats](https://packaging.python.org/en/latest/discussions/package-formats/)
  distinguishes sdists from wheels and documents archive inspection; compiled
  extension wheels are platform/interpreter dependent.
- [PyPA packaging flow](https://packaging.python.org/en/latest/flow/) separates
  source trees, build artifacts, upload, and installed environments.
- [Source distribution specification](https://packaging.python.org/en/latest/specifications/source-distribution-format/)
  defines modern sdist archive structure; project metadata still does not prove
  a particular produced archive's members.
- [GitHub workflow artifacts](https://docs.github.com/en/actions/concepts/workflows-and-actions/workflow-artifacts)
  distinguishes run artifacts from caches and explains job/run persistence.
- [PyPI file-reuse help](https://pypi.org/help/#file-name-reuse) documents that
  uploaded distribution filenames cannot be reused.
