# Tabulated EOS Lifetime and Loader

This leaf owns the Python-visible tabulated-EOS path, initialization, callable
methods, NN model loading, owned-memory cleanup, post-close behavior, and
extension/shared-library loader boundary. Authority is
`src/pyghl/eos.py`, `src/pyghl/__init__.py`, `csrc/pyghl_module.c`, `setup.py`,
and exact parent-pinned GRHayL objects; no runtime success is claimed here.

## Table path precedence

`src/pyghl/eos.py::_resolve_table_path(table)` applies this exact order:

1. Convert the string/path-like input to `Path` and expand `~`.
2. If that path is an existing file, return its resolved path. This direct path
   wins over `GRHAYL_EOS_TABLE_DIR`.
3. If it is missing and has a parent other than `.`, raise `FileNotFoundError`;
   there is no environment-directory fallback for a missing explicit path.
4. For a missing basename, read `GRHAYL_EOS_TABLE_DIR`. If nonempty, expand its
   `~` and test `<basename>`, `<basename>.h5`, then `<basename>.hdf5`; return the
   first existing file as a resolved path.
5. Otherwise raise `FileNotFoundError` explaining direct-path or environment
   options.

Suffixes are appended to the provided basename literally. For example, a
missing basename already ending in `.h5` causes fallback candidates including
that exact name before suffixed variants. The environment directory itself is
not resolved until an existing candidate is returned.

`initialize_tabulated_eos_functions_and_params(params, table, **kwargs)` only
adds this resolution layer, then calls the low-level extension initializer with
the resolved string and forwarded keyword arguments.

## Initialization flow and defaults

| Stage | Exact source behavior | Failure/state boundary |
| --- | --- | --- |
| Python facade | resolves table path and delegates | `FileNotFoundError` before C call |
| C parse/type gate | requires `Params`, UTF-8 table string, and known keywords; checks `access(path, R_OK)` | parser/`TypeError` or errno-derived `OSError` |
| Wrapper allocation | allocates `PyGHLTabulatedEOS`, zeroes embedded struct, sets `initialized = 0` | CPython allocation failure |
| Function dispatch | calls `ghl_initialize_eos_functions(ghl_eos_tabulated)` | pinned GRHayL assigns tabulated function pointers when configured with HDF5 |
| Table/model initialization | calls `ghl_initialize_tabulated_eos` with StellarCollapse table type, sound-speed cleaning false, requested NN flag, and configured bounds | non-success status becomes `GRHayLError`; no Python EOS returned |
| Success finalization | assigns `root_finding_precision`, then sets `initialized = 1` | returned wrapper owns table/model allocations |

Low-level defaults in `py_initialize_tabulated_eos_functions_and_params` are:

| Keyword | Default | Keyword | Default |
| --- | ---: | --- | ---: |
| `rho_atm` | `1e-12` | `rho_min` | `1e-12` |
| `rho_max` | `1e300` | `Ye_atm` | `0.5` |
| `Ye_min` | `0.05` | `Ye_max` | `0.5` |
| `T_atm` | `1e-2` | `T_min` | `1e-2` |
| `T_max` | `1e2` | `root_finding_precision` | `1e-10` |
| `enable_neural_net_c2p` | false |  |  |

Pinned `GRHayL/GRHayL_Core/initialize_eos.c::ghl_initialize_tabulated_eos`
reads the table through the assigned function pointer, constrains requested
bounds against table bounds, and initializes the embedded EOS parameters. When
`enable_neural_net_c2p` is true, pinned source also attempts to load the
`grhayl_nn_c2p` group from that EOS file during initialization; a load failure
frees table/model memory and returns an error. Setting the exposed boolean after
initialization only changes the field; the setter does not load a model.

On a wrapper-observed initialization error, current C source conditionally calls
`ghl_tabulated_free_memory`, translates status through `raise_ghl_error`, and
decrements the uninitialized wrapper. The embedded-NN load-failure path is
unsafe: pinned `ghl_initialize_tabulated_eos()` first calls
`ghl_tabulated_free_memory()`, freeing table arrays and performing model cleanup,
then the wrapper calls the same free function again. Pinned cleanup nulls model
and beta-equilibrium pointers but not the freed table-array pointers, so this
path can double-free table allocations and has undefined behavior. Do not enable
embedded-NN loading during initialization until cleanup has a single owner.

## Properties and callable methods

| Surface | Input | Return/mutation | Gates |
| --- | --- | --- | --- |
| `rho_min/max`, `Ye_min/max`, `T_min/max`, `table_T_min/max` | property read | Python float from embedded struct | no wrapper initialized guard |
| `root_finding_precision` | read/write property | converts/updates embedded double | no wrapper initialized guard; deletion/type rejected |
| `enable_neural_net_c2p` | read/write property | Python truth test/update embedded bool | no wrapper initialized guard |
| `tabulated_enforce_bounds_rho_Ye_T(rho, Ye, T)` | three doubles | clipped three-tuple | initialized guard plus explicit non-null pointer gate |
| `tabulated_enforce_bounds_rho_Ye_eps(rho, Ye, eps)` | three doubles | clipped three-tuple | initialized guard plus explicit non-null pointer gate |
| `tabulated_compute_P_from_T` | `(rho, Ye, T)` | pressure float | initialized guard; GRHayL status translation |
| `tabulated_compute_eps_from_T` | `(rho, Ye, T)` | energy float | same |
| `tabulated_compute_cs2_from_T` | `(rho, Ye, T)` | sound-speed-squared float | same |
| `tabulated_compute_P_eps_from_T` | `(rho, Ye, T)` | `(P, eps)` | same |
| `tabulated_compute_P_eps_S_from_T` | `(rho, Ye, T)` | `(P, eps, entropy)` | same |
| `tabulated_compute_T_from_eps` | `(rho, Ye, eps)` | temperature float | same |
| `tabulated_compute_P_T_from_eps` | `(rho, Ye, eps)` | `(P, T)`; wrapper seeds inversion `T` with `table_T_max` | same |
| `load_nn_c2p_hdf5(model_path)` | path parsed as C string | `None`; successful pinned load replaces prior model | initialized guard, readable-path gate, GRHayL status translation |
| `close()` | none | `None`; conditionally frees and clears initialized flag | see lifecycle below |

The seven compute/inversion wrappers call their GRHayL function pointers after
the initialized gate but do not individually test those pointers for null. Their
source prerequisite is successful tabulated pointer initialization. Only the
two bounds wrappers have explicit null-pointer checks and raise a dedicated
`RuntimeError` if missing.

## Standalone NN loading

`load_nn_c2p_hdf5` does not mutate the EOS table file. It requires an initialized
wrapper and a readable model path, then calls pinned
`ghl_c2p_nn_load_hdf5`. Pinned
`GRHayL/Con2Prim/Tabulated/neural_network_guess/c2p_nn_load_from_eos_hdf5.c`
opens the standalone HDF5 read-only, reads and checks required scalar/array
shapes, validates the candidate model, and only on success frees the prior model
and installs the candidate. On failure it frees the candidate and returns a
status translated to `GRHayLError` by pyghl.

Loading alone does not change `enable_neural_net_c2p`. `nn_c2p_guess` and
`nn_c2p_guess_x` separately require initialized EOS state and non-null
`eos.c2p_nn`; otherwise pyghl raises `GRHayLError`. Model schema, EOS embedding,
installation, and artifact validity belong to
[model/EOS HDF5 lifecycle](../nn/model-eos-hdf5-lifecycle.md).

## Ownership, `close`, and deallocation

After successful initialization, the wrapper owns GRHayL table arrays and any
loaded C2P NN model. Pinned
`GRHayL/EOS/Tabulated/NRPyEOS_free_memory.c::NRPyEOS_free_memory` frees table
arrays, beta-equilibrium allocations, and the C2P model, then nulls the model
pointer.

`close()` and `eos_dealloc` share this condition: only when wrapper
`initialized` is true and `ghl_tabulated_free_memory` is non-null do they call
the free function and clear the flag. Consequences:

- after a normal successful free, repeated `close()` returns `None` without a
  second free;
- deallocation after such a close skips upstream free and releases the Python
  object through `tp_free`;
- if the wrapper says initialized but the free pointer is null, `close()`
  returns `None` without clearing the flag, and deallocation likewise cannot
  invoke upstream free; current source reports no Python error for this case.

After a normal close, every EOS instance method that calls
`eos_ensure_initialized` raises `RuntimeError("TabulatedEOS is not
initialized.")`. Module calls `guess_primitives`, the two tabulated C2P calls,
and NN guesses also apply that guard. Property getters/setters and `repr` do not;
they remain source-callable against residual embedded scalar fields and must not
be treated as access to a live table. Use-after-close property values are not
runtime compatibility or numerical evidence.

## Failure layers

| Layer | Representative failures | Python-visible boundary |
| --- | --- | --- |
| Python validation/path resolution | missing direct path/basename, wrong path-like use | `FileNotFoundError` or Python conversion error in `eos.py` |
| Extension validation | wrong `Params`, bad arguments, unreadable table/model file | `TypeError`, parser exception, or errno-derived `OSError` |
| GRHayL/HDF5/table/model | disabled HDF5, open/read/shape/bounds/allocation/model validation errors | symbolic/integer `GRHayLError` from returned status |
| Function-pointer prerequisite | explicit missing bounds pointer; compute pointers rely on successful initialization | `RuntimeError` for checked bounds pointers; no stronger unchecked-pointer claim |
| Post-close | guarded EOS/module calls | `RuntimeError` from `eos_ensure_initialized` |
| Python extension import | `_pyghl` import raises `ImportError` | stored original error; bindings absent from conditional facade |
| Dynamic loader | original `ImportError` text names missing/unloadable dependency such as `libghl` | contextual chained `ImportError` from `require_bindings` |

`src/pyghl/__init__.py` catches `ImportError` around its extension import,
stores it in `_BINDINGS_IMPORT_ERROR`, leaves `_BINDINGS_AVAILABLE` false, and
still imports `nn`. `require_bindings()` preserves the original text and cause.
If that text contains `libghl`, it computes a diagnostic library directory from
`GRHAYL_DIR/build/lib` or the repository checkout fallback, suggests
`LD_LIBRARY_PATH` for local/editable use, and recommends rebuilding/reinstalling
regular installs. Other loader failures receive a generic rebuild/inspect
message. Diagnostic advice is not remediation execution.

## Build and packaging configurations

| Mode | Configuration says | Strongest source-only evidence |
| --- | --- | --- |
| default source/editable build | `setup.py` selects `extern/GRHayL`, configures if `Makefile` or `build/` is missing, runs `make ... grhayl`, links against `build/lib` | `implemented`/`selected`; no compile/link/import claim |
| external GRHayL build | nonempty `GRHAYL_DIR` selects another checkout for configure/include/library inputs | build root `selected`; parent-pin claims no longer describe that external build |
| inplace extension | always adds platform-local rpath and additionally selected `build/lib` rpath | link arguments `selected`; loader success unproved |
| non-inplace build/wheel input | copies enumerated existing `libghl` names beside extension; uses `$ORIGIN` on non-macOS or `@loader_path` on macOS; rewrites macOS library IDs/load paths | `package-selected`; inspect named wheel for `packaged` |
| sdist selection | `MANIFEST.in` includes README, C source, and recursive Python/typed/HDF5 files | `package-selected`; does not prove GRHayL/build inputs or actual sdist contents |
| wheel workflow | selected matrices build and run `import pyghl; pyghl.require_bindings()` | workflow text is `selected`; only a named hosted run proves execution |

`pyproject.toml` package-data globs for shared libraries and tracked models are
also `package-selected`. A copied source-tree library, built extension, sdist,
wheel, editable install, and external checkout are distinct evidence objects.
Do not infer one from another.

## Artifact boundary

| Artifact | Producer | Consumer | Location | Ownership | Mutation/cleanup | Strongest baseline proof |
| --- | --- | --- | --- | --- | --- | --- |
| EOS HDF5 table | external/user workflow | GRHayL table initializer; optional embedded NN loader | caller path | user-owned | opened/read by this binding path; wrapper `close` frees memory, not file | source consumer implemented; no artifact inspected here |
| standalone NN HDF5 | training/export/user workflow | `load_nn_c2p_hdf5` and pinned loader | caller path | user-owned | read-only load; successful load replaces in-memory prior model | source consumer implemented |
| in-memory table/model | pinned GRHayL initializer/loader | EOS and C2P calls | owned embedded C struct/pointers | wrapper-owned | `close`/deallocator free under pointer gate | source cleanup implemented |
| `_pyghl` extension | build backend | Python import machinery | package directory | build/package-managed | rebuilt/reinstalled, not edited | package-selected only |
| `libghl` shared library | GRHayL build; non-inplace copy step | dynamic loader/extension | selected build lib or beside extension | build/package-managed | rebuild/copy; platform loader resolves | package-selected only |

## Proof gaps and change impact

No root test directly covers path precedence, initializer defaults, function
pointer gates, standalone model replacement, `close`, deallocation, post-close
properties, or loader diagnostics. `examples/eos_smoke.py` is an operational
consumer with a user-specific table path, not an assertion. Upstream tests do
not prove Python ownership. See [test map](../test-map.md).

- `src/pyghl/eos.py`: review path precedence and public API map.
- EOS wrapper initializer/method/deallocator changes: review this page,
  [extension surface](extension-surface-and-errors.md), pinned declarations,
  and direct wrapper proof.
- `setup.py`, package metadata, or workflow changes: review mode-specific
  selection without upgrading evidence state absent artifact/run inspection.
- GRHayL EOS/NN free/load/pointer changes: inspect parent-pinned object paths and
  update pyghl integration only where boundary behavior changes.
