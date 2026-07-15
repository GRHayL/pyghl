# Bindings Hub

This hub routes the CPython extension's seven wrapper families, module calls,
error boundary, and tabulated-EOS lifecycle. `csrc/pyghl_module.c` owns wrapper
semantics; parent-pinned GRHayL declarations/definitions own delegated C
behavior and numerical algorithms.

## Start by question

| Query/task | Canonical owner | Primary authority | Proof/workflow route |
| --- | --- | --- | --- |
| Which type, property, function, constant, mutation, or return is registered? | [Extension surface and errors](extension-surface-and-errors.md) | `csrc/pyghl_module.c` type/getset/method tables and `PyInit__pyghl` | [Public API map](../public-api-map.md) |
| Why is a top-level binding absent or import failing? | [Tabulated EOS lifetime and loader](tabulated-eos-lifetime-and-loader.md) | `src/pyghl/__init__.py` import block and `require_bindings` | [Build/package/release/CI](../build-package-release-ci.md) |
| How is an EOS table path chosen and table memory released? | [Tabulated EOS lifetime and loader](tabulated-eos-lifetime-and-loader.md) | `src/pyghl/eos.py`; C EOS initializer/methods/deallocator | [GRHayL integration](../integration/grhayl-submodule.md) |
| Which GRHayL file owns delegated behavior? | [GRHayL integration](../integration/grhayl-submodule.md) | parent gitlink, then pinned public header and definition | pinned nested `AGENTS.md` |
| What directly proves wrapper behavior? | [Test map](../test-map.md) | exact repository assertions and named execution | coverage gaps remain distinct from upstream tests/examples |

## Seven wrapper families

| Python type | Embedded C value | Creation routes | Public surface summary | Owner |
| --- | --- | --- | --- | --- |
| `Params` | `ghl_parameters` | `initialize_params`; type has no `tp_new` | read-only routine/configuration getters | [Extension surface and errors](extension-surface-and-errors.md) |
| `TabulatedEOS` | `ghl_eos_parameters` plus wrapper `initialized` flag | low-level initializer; type has no `tp_new` | bounds/configuration getters, two setters, interpolation/bounds methods, NN load, `close` | [EOS lifetime](tabulated-eos-lifetime-and-loader.md) |
| `Primitive` | `ghl_primitive_quantities` | `initialize_primitives`; type also uses `PyType_GenericNew` | scalar/vector read-write fields | [Extension surface and errors](extension-surface-and-errors.md) |
| `Conservative` | `ghl_conservative_quantities` | returned by conversion helpers; type also uses `PyType_GenericNew` | scalar/vector read-write fields | [Extension surface and errors](extension-surface-and-errors.md) |
| `Metric` | `ghl_metric_quantities` | `initialize_metric`; type also uses `PyType_GenericNew` | read-only scalar/vector/matrix fields | [Extension surface and errors](extension-surface-and-errors.md) |
| `ADMAux` | `ghl_ADM_aux_quantities` | `compute_ADM_auxiliaries`; type also uses `PyType_GenericNew` | opaque wrapper; representation only | [Extension surface and errors](extension-surface-and-errors.md) |
| `Diagnostics` | `ghl_con2prim_diagnostics` | constructor, `initialize_diagnostics`, and C2P calls | mutable flags/routine/count; read-only backup tuple | [Extension surface and errors](extension-surface-and-errors.md) |

Creation-route presence is not proof that direct construction creates a
scientifically meaningful value. Use the named initializer/producer required by
the consuming call and retain exact prerequisites.

## Representative Python-to-parent-pin traces

Use the dynamically resolved parent gitlink as described by
[GRHayL integration](../integration/grhayl-submodule.md). Never substitute a
different nested checkout for pinned object evidence.

| Python journey | Python/C wrapper trace | Parent-pinned declaration | Parent-pinned definition/assignment | Established state |
| --- | --- | --- | --- | --- |
| `pyghl.initialize_params(...)` | conditional import in `src/pyghl/__init__.py`; `module_methods`; `py_initialize_params` | `GRHayL/include/ghl.h::ghl_initialize_params` | `GRHayL/GRHayL_Core/initialize_params.c::ghl_initialize_params` | `exported` conditionally; C call `registered` and `implemented` |
| `pyghl.initialize_metric(...)` then `compute_ADM_auxiliaries` | two `module_methods` entries; `py_initialize_metric`; `py_compute_ADM_auxiliaries` | `GRHayL/include/ghl.h::ghl_initialize_metric`, `ghl_compute_ADM_auxiliaries` | `GRHayL/GRHayL_Core/initialize_metric.c`, `compute_ADM_auxiliaries.c` | source call chain implemented |
| `pyghl.compute_conservs(metric, aux, prims)` | facade import; `module_methods`; `py_compute_conservs`; exact type gates | `GRHayL/include/ghl_con2prim.h::ghl_compute_conservs` | `GRHayL/Con2Prim/compute_conservs.c::ghl_compute_conservs` | source call chain implemented |
| `pyghl.limit_v_and_compute_u0(params, metric, prims)` | facade import; `py_limit_v_and_compute_u0`; returns wrapper-converted flag | `GRHayL/include/ghl.h::ghl_limit_v_and_compute_u0` | `GRHayL/GRHayL_Core/limit_v_and_compute_u0.c::ghl_limit_v_and_compute_u0` | source mutation/error/return wiring implemented |
| `pyghl.eos.initialize_tabulated_eos_functions_and_params(...)` | Python path resolver; low-level `_pyghl` initializer; `ghl_initialize_eos_functions`; table initializer | `GRHayL/include/ghl.h::ghl_initialize_tabulated_eos`; tabulated pointers in `GRHayL/include/ghl_eos_functions.h` | `GRHayL/GRHayL_Core/initialize_eos.c`; `GRHayL/EOS/Tabulated/NRPyEOS_initialize_tabulated_functions.c` | source lifecycle wiring implemented; HDF5/table/runtime still gated |
| `eos.tabulated_compute_P_from_T(...)` | `eos_methods`; initialized guard; wrapper calls function pointer; translates status | `GRHayL/include/ghl_eos_functions.h::ghl_tabulated_compute_P_from_T` | assignment to `NRPyEOS_P_from_rho_Ye_T` in tabulated initializer; interpolator under `GRHayL/EOS/Tabulated/interpolators/` | declared/assigned/called at pin; no numerical execution claimed |
| `eos.load_nn_c2p_hdf5(...)` then `pyghl.nn_c2p_guess(...)` | readable-path gate; load status translation; loaded-model gate; guess call | `GRHayL/include/ghl_con2prim.h::ghl_c2p_nn_load_hdf5`, `ghl_c2p_nn_guess` | neural-network guess loader and `c2p_nn_guess_x.c` under `GRHayL/Con2Prim/Tabulated/neural_network_guess/` | source-present and implemented; actual model load/guess unproved |

## Evidence boundary

- C tables and facade imports establish source registration/export routes only.
- Exact upstream headers and definitions establish parent-pin implementation,
  not Python wrapper execution or numerical correctness.
- `examples/eos_smoke.py` and README examples are operational consumers, not
  direct test evidence.
- Root tests do not directly exercise these wrapper types, lifetime paths, or
  translated GRHayL errors; see [test map](../test-map.md).
- A named compiled extension, loader-visible `libghl`, EOS table, and optional
  NN model are prerequisites for relevant runtime evidence.

## Change impact

- Registration/getset/wrapper parsing change: update extension owner and public
  API map; review direct wrapper proof.
- EOS ownership, close, table path, function-pointer, or loader change: update
  EOS lifetime owner and build integration.
- Delegated GRHayL signature/status change: retrace exact parent-pinned header
  and definition before updating wrapper claims.
