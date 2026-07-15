# GRHayL Submodule and Integration Boundary

This page owns pyghl's boundary with GRHayL: repository pinning, source
authority, build selection, include/link inputs, and safe upstream tracing. It
does not own GRHayL algorithms; consult the parent-pinned upstream objects and
their nested `AGENTS.md` for those contracts.

## Read first

- `.gitmodules` records submodule name, path, fetch URL, and a `main` branch
  hint.
- The parent `HEAD` tree records `extern/GRHayL` as a gitlink. That entry, not
  the nested working-tree `HEAD`, identifies durable upstream authority for the
  parent revision.
- `extern/GRHayL/AGENTS.md` and its routed upstream pages govern work below the
  submodule. Read them from the parent-pinned object when validating durable
  claims.
- `setup.py` owns pyghl's build and link seam. [Build/package/release/CI](../build-package-release-ci.md)
  owns the broader package workflow.

## Safe parent-pin inspection

Resolve the pin dynamically and ask the nested repository's object database for
content:

```bash
pin=$(git ls-tree HEAD extern/GRHayL | awk '{print $3}')
test -n "$pin"
git -C extern/GRHayL cat-file -e "$pin^{commit}"
git -C extern/GRHayL show "$pin:AGENTS.md"
git -C extern/GRHayL ls-tree -r --name-only "$pin"
git -C extern/GRHayL grep -n 'ghl_initialize_params' "$pin" -- GRHayL/
```

Use `git -C extern/GRHayL show "$pin:path"` for exact files. Never reset,
update, checkout, stage, or edit the nested repository merely to inspect the
parent pin. A nested working tree may legitimately point elsewhere; it is local
state, not substitute evidence for paths or symbols at the pin. Permanent KB
pages name paths and symbols, not a literal object ID.

## Configuration and authority

| Concern | Source-present contract | Authority/evidence state |
| --- | --- | --- |
| Submodule mapping | `.gitmodules` maps `extern/GRHayL` to the official GRHayL repository and records a branch hint | `declared`; branch hint does not replace parent gitlink |
| Durable upstream selection | parent tree contains a mode-`160000` gitlink at `extern/GRHayL` | `selected`; inspect nested objects at that object name |
| Upstream navigation | parent-pinned `AGENTS.md` routes GRHayL source, tests, public headers, and upstream KB | router only; exact pinned source remains authority |
| Alternate build source | `setup.py::_default_grhayl_root` uses nonempty `GRHAYL_DIR`; otherwise `extern/GRHayL` | `selected` at build-configuration evaluation; no build implied |
| Configure arguments | `setup.py::_run_make_grhayl` starts with `--prefix=.` and appends shell-split `GRHAYL_CONFIGURE_ARGS` | `selected`; actual configure execution requires run evidence |
| Configure gate | missing `Makefile` or `build/` triggers the selected checkout's `configure`; missing script raises `RuntimeError` | `implemented`; no configured checkout implied |
| Library build | `_run_make_grhayl` invokes `make -C <selected-root> grhayl` | `implemented`; `compiled` requires a named successful build |
| Headers | extension includes `<selected-root>/GRHayL/include` and `<selected-root>/Unit_Tests/data_gen` | build input selected in `setup.py` |
| Link input | extension searches `<selected-root>/build/lib` and links library name `ghl` | build/link selection only; no library or successful link implied |
| Loader paths | build extension supplies local platform rpath; inplace additionally supplies selected build library path | link arguments selected; see [EOS loader](../bindings/tabulated-eos-lifetime-and-loader.md) |
| Non-inplace build copy | When `BuildExt.run` observes `not self.inplace`, it copies enumerated existing `libghl` Linux/macOS names beside the extension and rewrites macOS IDs/load paths | `implemented`/`package-selected`; frontend labels do not prove the predicate, so establish its actual value and inspect the named build output or wheel as claimed |

`GRHAYL_DIR` changes build authority from the parent-pinned submodule to an
external checkout for that invocation. Claims about such a build must name that
checkout/configuration; they must not be presented as parent-pin behavior.

## Parent-pin traces

Each trace starts in pyghl, crosses a pinned public declaration, and ends at a
pinned definition or function-pointer assignment. These prove source wiring,
not numerical correctness or runtime execution.

| Python/C entry | pyghl call site | Parent-pinned declaration | Parent-pinned definition/assignment | Strongest baseline state |
| --- | --- | --- | --- | --- |
| `initialize_params` | `csrc/pyghl_module.c::py_initialize_params` calls `ghl_initialize_params` | `GRHayL/include/ghl.h::ghl_initialize_params` | `GRHayL/GRHayL_Core/initialize_params.c::ghl_initialize_params` | `implemented` at pin |
| `compute_conservs` | `csrc/pyghl_module.c::py_compute_conservs` | `GRHayL/include/ghl_con2prim.h::ghl_compute_conservs` | `GRHayL/Con2Prim/compute_conservs.c::ghl_compute_conservs` | `implemented` at pin |
| tabulated initialization | `csrc/pyghl_module.c::py_initialize_tabulated_eos_functions_and_params` | `GRHayL/include/ghl.h::ghl_initialize_tabulated_eos` | `GRHayL/GRHayL_Core/initialize_eos.c::ghl_initialize_tabulated_eos` | `implemented` at pin |
| pressure interpolation | `csrc/pyghl_module.c::eos_tabulated_compute_P_from_T` calls function pointer | `GRHayL/include/ghl_eos_functions.h::ghl_tabulated_compute_P_from_T` | `GRHayL/EOS/Tabulated/NRPyEOS_initialize_tabulated_functions.c::NRPyEOS_initialize_tabulated_functions` assigns `NRPyEOS_P_from_rho_Ye_T` | `declared` plus assigned at pin |
| velocity limit | `csrc/pyghl_module.c::py_limit_v_and_compute_u0` | `GRHayL/include/ghl.h::ghl_limit_v_and_compute_u0` | `GRHayL/GRHayL_Core/limit_v_and_compute_u0.c::ghl_limit_v_and_compute_u0` | `implemented` at pin |
| multi-method C2P | `csrc/pyghl_module.c::py_tabulated_con2prim_multi_method` | `GRHayL/include/ghl_con2prim.h::ghl_con2prim_tabulated_multi_method` | `GRHayL/Con2Prim/con2prim_multi_method.c::ghl_con2prim_tabulated_multi_method` | `implemented` at pin |
| standalone NN load | `csrc/pyghl_module.c::eos_load_nn_c2p_hdf5` | `GRHayL/include/ghl_con2prim.h::ghl_c2p_nn_load_hdf5` | `GRHayL/Con2Prim/Tabulated/neural_network_guess/c2p_nn_load_from_eos_hdf5.c::ghl_c2p_nn_load_hdf5` | `implemented` at pin; HDF5 configuration remains a gate |

For tabulated function pointers, initialization crosses
`ghl_initialize_eos_functions(ghl_eos_tabulated)` before the table initializer.
The pinned `GRHayL/GRHayL_Core/initialize_eos.c` dispatch and
`GRHayL/EOS/Tabulated/NRPyEOS_initialize_tabulated_functions.c` assignments are
therefore part of prerequisite tracing. [Tabulated EOS lifetime and loader](../bindings/tabulated-eos-lifetime-and-loader.md)
owns the Python-visible lifecycle.

## Upstream-only surfaces

The pinned GRHayL public headers declare much more than pyghl registers or
exports. Flux/source, induction, atmosphere, reconstruction, neutrino, hybrid
EOS, and many additional tabulated/Con2Prim routines are `upstream-only` unless
a pyghl registration/export trace exists in [public API map](../public-api-map.md).
Their presence in a linked library does not create Python exposure.

## Change impact

- `.gitmodules` or parent gitlink: review this page, build selection, all pinned
  traces, packaging, and any wrapper claim whose upstream declaration changed.
- `setup.py` GRHayL selection/include/link/copy logic: review this page,
  [build/package/release/CI](../build-package-release-ci.md), and
  [tabulated EOS loader](../bindings/tabulated-eos-lifetime-and-loader.md).
- Wrapper use of a new GRHayL symbol: add declaration and definition/assignment
  traces before claiming exposure.
- Upstream-only algorithm change: edit upstream under its nested instructions;
  parent KB changes only when pyghl integration or routing changes.

## External Ground Truth

- [Git submodule guide](https://git-scm.com/docs/gitsubmodules) defines a
  submodule as an embedded repository with independent history, explains the
  superproject gitlink and `.gitmodules`, and states that the gitlink records
  the expected submodule commit.
- [`git ls-tree`](https://git-scm.com/docs/git-ls-tree) documents inspection of
  entries in a tree object.
- [`git cat-file`](https://git-scm.com/docs/git-cat-file) documents object
  existence/type/content inspection used by the pin-safe commands above.
