# Neural-Network Inference and Export

This page owns PyTorch bundle inference, public GRHayL-backed guesses, C-header export, and header-to-HDF5 conversion. Authority is `src/pyghl/_nn_infer.py`, `src/pyghl/_nn_train.py`, `src/pyghl/nn.py`, `csrc/pyghl_module.c`, and `src/pyghl/nn_c2p/header_to_hdf5.py`; representation production alone is not Python/C parity proof.

## Representation Matrix

| Representation | Producer | Reader/consumer | Evidence boundary |
| --- | --- | --- | --- |
| PyTorch inference bundle | `save_inference_bundle()` | `load_inference_bundle()`, `_prepare_bundle_on_device()` | Source save/load agreement only |
| Prepared in-memory bundle | `_prepare_bundle_on_device()` | `cons_to_x_guess()` | Process-local, device-mutating cache |
| Generated C header | `export_to_c_header()` | C compilation or `header_to_payload()` | Text representation; no compiled/executed proof |
| Standalone HDF5 | `export_to_hdf5()` or header converter | Python lifecycle and parent-pinned GRHayL loader | Schema source trace; compatibility separately bounded |
| Loaded C model | `TabulatedEOS.load_nn_c2p_hdf5()` or embedded EOS initialization | `nn.guess()` / `nn.guess_x()` | Requires compiled bindings and successful GRHayL load |

## Bundle Save and Load

`save_inference_bundle()` unwraps a compiled model through `_orig_mod`, moves state tensors and statistics to CPU, and calls `torch.save`. Bundle keys are:

- `arch`: input, hidden, hidden-stage, and output dimensions;
- `state_dict`: model parameters;
- `ft_stats`: transform kind, epsilon, and q/s indices;
- `x_stats` and `y_stats`: float32 tensors used for scaling and decode.

`load_inference_bundle(path, map_location=...)` calls `torch.load(..., weights_only=True)`. A Torch version lacking that keyword becomes a `RuntimeError`; a non-dict becomes `ValueError`. It reconstructs `TinyMLP_Logit`, loads state, sets evaluation mode, reconstructs transform metadata, and returns a tuple. Missing keys, incompatible shapes, and unsupported serialized objects surface from dictionary/model/Torch operations; no schema version field or custom migration exists.

## Python Prediction Path

`_prepare_bundle_on_device()` accepts the returned tuple or a dict, moves model and x statistics to selected device, mutates `ft_stats.kind` onto that device, normalizes scalar `y_eps`/`width_tiny`, moves output arrays, and allocates one `(1,4)` scratch tensor. Calling `cons_to_x_guess()` with an already prepared dict reuses scratch; other inputs are prepared using explicit device or current model device.

Prediction rejects nonfinite or too-small `1+q`, writes `q,r,s,t` into scratch, applies configured identity/log10 transforms, robust clipping/min-max scaling, model/HardTanh logits, sigmoid, and output decode. Bounded-x decode uses `1+q-s + clamp(y01)*(1+q)`; linear decode uses `out_lo + y01/out_invrng`; log-linear exponentiates that result. Returned scalar is moved to CPU.

Prepared dict reuse is mutable and shares scratch/model/stat objects; source provides no thread-safety contract. `pick_inference_device()` exists but `cons_to_x_guess()` otherwise follows model or explicit device.

## Public C Guess Path

`nn.guess(eos,q,r,s,t)` requires compiled bindings and calls registered
`_pyghl.nn_c2p_guess`; `guess_x()` returns its scalar. The C wrapper requires a
`TabulatedEOS`, parses four floats, raises `GRHayLError` when no `c2p_nn` is
loaded, calls parent GRHayL `ghl_c2p_nn_guess`, and returns the resulting `.x`.
That GRHayL call returns a guess struct, not an error status, so the wrapper
performs no post-call status translation. `nn.nn_initial_guess()` computes
`q,r,s,t` from conservative state, invokes this C path, clamps x to bounds, and
mutates primitive fields through EOS/native helpers.

This is independent from `_nn_infer.cons_to_x_guess()`. Similar source transformations do not prove bitwise, tolerance, or failure-mode parity. No root test compares them.

## C-Header Export

`export_to_c_header()` requires a `TinyMLP_Logit`-like model, normalizes compile/module key prefixes, extracts float32 weights/stats, checks dimension lengths and finiteness, and serializes macros plus `static const` arrays. It includes input/output metadata, all layer arrays, and optional audit comments.

Header guard is sanitized uppercase basename plus first 10 uppercase hex digits of `sha1(basename)`. This is a generated product identifier reducing same-name guard collisions; it does not hash header content, authenticate the model, or represent KB freshness.

Path is opened with text `w` only after in-memory validation, so an existing header is truncated. Write failure can leave partial text. Export neither compiles header nor compares C output with Python.

## Header-to-HDF5 Conversion

`header_to_payload()` uses regular expressions and `ast.literal_eval` to parse expected macros, arrays, and optional audit comment. It casts values to NumPy int32/float32 and synthesizes empty hidden arrays when `n_hidden==1`. `write_hdf5()` maps that data into standard HDF5 groups; optional `--eos-file` adds source EOS metadata.

Converter trusts recognized textual declarations, does not run a C preprocessor, and does not verify guard SHA-1, array shapes, finiteness, or inference equivalence before HDF5 writing. Root attribute `source_header` records provided path string as product provenance.

| Artifact | Producer | Consumer | Location | Tracked/user-owned | Mutation/overwrite | Guard/cleanup | Strongest proof |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Inference `.pt` | bundle saver | bundle loader | Caller path | User-owned/generated | `torch.save` may overwrite | Restricted loader; no temp/rollback | `implemented` |
| C header `.h` | header exporter/checkpoint export | C compiler or converter | Caller/checkpoint path | User-owned/generated | Text `w` truncates | Pre-write dimension/finite checks; no rollback | `implemented` |
| Converted HDF5 | header converter | HDF5 lifecycle/GRHayL | CLI output | User-owned/generated | HDF5 writer uses `w` | Parser exceptions before/during write; no temp/rollback | `implemented` |
| Prepared device bundle | inference helper | scalar predictor | Memory/device | Ephemeral | Mutates model/transform device and scratch | Width check only at prediction | `implemented` |

## Evidence and Gaps

Save/load keys, transforms, decode, C wrapper, header text, and conversion are source-traced. No root tests execute bundle load/prediction, C-header compilation, conversion, standalone loader, or Python/C parity. No bundle/header/model was created or executed for KB work.

## Change Impact

Architecture, key, transform, output-kind, or array-order changes require [Training and checkpoints](training-and-checkpoints.md), [HDF5 lifecycle](model-eos-hdf5-lifecycle.md), converter, C wrapper, parent-pinned loader/validator/inference, and new parity fixtures. Serialization changes require trusted-input and migration review.

## External Ground Truth

- [PyTorch `torch.save`](https://docs.pytorch.org/docs/stable/generated/torch.save.html) defines object serialization used by bundle production.
- [PyTorch `torch.load`](https://docs.pytorch.org/docs/stable/generated/torch.load.html) documents `weights_only`, unpickling risk, and device remapping.
- [Python `hashlib`](https://docs.python.org/3/library/hashlib.html) documents SHA-1 and known collision weaknesses; project guard use is non-security identification.
