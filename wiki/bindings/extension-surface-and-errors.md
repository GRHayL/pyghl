# Extension Surface, Ownership, and Errors

This leaf owns source-visible behavior of `csrc/pyghl_module.c`: seven static
wrapper types, fields, module calls/constants, mutation and return behavior,
input validation, diagnostics, CPython ownership, and error translation.
Numerical algorithms remain owned by parent-pinned GRHayL source routed through
[GRHayL integration](../integration/grhayl-submodule.md).

## Registration boundary

`PyInit__pyghl` calls `PyType_Ready` for all seven types, creates `_pyghl` from
`pyghl_module`, creates `pyghl.GRHayLError` as a `RuntimeError` subclass, and
adds type objects and constants. `module_methods`, each `PyGetSetDef`,
`eos_methods`, and the module initializer establish `registered` source wiring.
Conditional top-level `exported` wiring belongs to
[public API map](../public-api-map.md); neither state proves import/execution.

Ten integer constants map directly to pinned `ghl_con2prim_id_t` enumerators:
`C2P_NONE`, `C2P_NOBLE2D`, `C2P_NOBLE1D`, `C2P_NOBLE1D_ENTROPY`,
`C2P_NOBLE1D_ENTROPY2`, `C2P_FONT1D`, `C2P_PALENZUELA1D`,
`C2P_PALENZUELA1D_ENTROPY`, `C2P_NEWMAN1D`, and
`C2P_NEWMAN1D_ENTROPY`.

## Wrapper types and fields

| Type | Construction and representation | Get/set contract |
| --- | --- | --- |
| `Params` | `initialize_params` allocates/initializes; no `tp_new`; repr prints configuration | read-only `main_routine`, `backup_routines` (three-int tuple), `evolve_entropy`, `evolve_temp`, `calc_prim_guess`, `psi6threshold`, `max_lorentz_factor`, `lorenz_damping_factor` |
| `TabulatedEOS` | low-level table initializer allocates/owns state; no `tp_new`; custom deallocator; repr reads bounds | read-only `rho_min`, `rho_max`, `Ye_min`, `Ye_max`, `T_min`, `T_max`, `table_T_min`, `table_T_max`; read-write `root_finding_precision`, `enable_neural_net_c2p`; full lifecycle in [EOS leaf](tabulated-eos-lifetime-and-loader.md) |
| `Primitive` | initializer producer plus `PyType_GenericNew`; repr prints selected scalars | read-write `rho`, `press`, `eps`, `u0`, `vU`, `BU`, `Y_e`, `temperature`, `entropy` |
| `Conservative` | producer functions plus `PyType_GenericNew`; repr prints selected scalars | read-write `rho`, `tau`, `Y_e`, `SD`, `entropy` |
| `Metric` | initializer producer plus `PyType_GenericNew`; repr prints selected scalars | read-only `lapse`, `lapseinv`, `detgamma`, `sqrt_detgamma`, `betaU`, `gammaDD`, `gammaUU` |
| `ADMAux` | `compute_ADM_auxiliaries` producer plus `PyType_GenericNew` | no public fields/methods; repr is `ADMAux()` |
| `Diagnostics` | custom `diagnostics_new`, initializer producer, or C2P call; all run `initialize_py_diagnostics`; repr prints selected fields | read-write `tau_fix`, `Stilde_fix`, `speed_limited`, `which_routine`, `n_iter`; read-only three-bool `backup` tuple |

Scalar setters use `PyFloat_AsDouble`; deleting them is rejected. Vector setters
`vU`, `BU`, and `SD` require a sequence of exactly three values and convert each
with the same double parser. Vector getters return tuples; metric matrix getters
return a three-by-three tuple. Bool setters use Python truth testing. Diagnostics
integer setters use `PyLong_AsLong`, then cast to the upstream enum or C `int`;
the wrapper contains no explicit enum-range or `int`-range gate.

Generic construction is source-present, but consumers requiring initialized
GRHayL invariants should use named producers. This page makes no runtime or
scientific-validity claim for directly created generic wrappers.

## Module functions

| Registered call | Validation/defaults | Return and visible mutation | Delegated pinned symbol |
| --- | --- | --- | --- |
| `initialize_params(...)` | optional main integer; backup `None` or exactly three integer keys; defaults: none main/backups, entropy false, temperature/guess true, `1e100`, `100.0`, `0.0` | new `Params` | `ghl_initialize_params` |
| `initialize_tabulated_eos_functions_and_params(...)` | exact `Params`, readable table path, optional table/bound keywords | new owned `TabulatedEOS`; [EOS leaf](tabulated-eos-lifetime-and-loader.md) | `ghl_initialize_eos_functions`, `ghl_initialize_tabulated_eos` |
| `initialize_metric(...)` | ten parsed doubles | new `Metric` | `ghl_initialize_metric` |
| `compute_ADM_auxiliaries(metric)` | exact wrapper family/subtype | new `ADMAux` | `ghl_compute_ADM_auxiliaries` |
| `initialize_primitives(...)` | twelve parsed doubles | new `Primitive` | `ghl_initialize_primitives` |
| `initialize_diagnostics()` | no arguments | new initialized `Diagnostics` | `ghl_initialize_diagnostics` |
| `compute_conservs(metric, metric_aux, prims)` | exact wrapper families/subtypes | new `Conservative`; inputs passed by pointer | `ghl_compute_conservs` |
| `undensitize_conservatives(psi6, cons)` | double plus conservative wrapper | new `Conservative`; input wrapper remains separate | `ghl_undensitize_conservatives` |
| `compute_SU_Bsq_Ssq_BdotS(metric, cons, prims)` | exact wrapper families/subtypes | `(SU, B_squared, S_squared, BdotS)`, with `SU` a three-tuple | `ghl_compute_SU_Bsq_Ssq_BdotS` |
| `limit_v_and_compute_u0(params, metric, prims)` | exact wrapper families/subtypes | mutates `prims`; returns `bool speed_limited`; translates status | `ghl_limit_v_and_compute_u0` |
| `limit_utilde_and_compute_v(params, metric, utildeU, prims)` | exact wrappers plus three-value vector | mutates `prims`; returns upstream bool | `ghl_limit_utilde_and_compute_v` |
| `guess_primitives(params, eos, metric, cons)` | exact wrappers; EOS must be initialized | new `Primitive` | `ghl_guess_primitives` |
| `tabulated_Palenzuela1D_energy(params, eos, metric, metric_aux, cons, prims[, diagnostics])` | exact wrappers; initialized EOS; optional diagnostics | upstream receives mutable `prims` and diagnostics; returns new diagnostics or same supplied object with incremented return reference; translates status | `ghl_tabulated_Palenzuela1D_energy` |
| `tabulated_con2prim_multi_method(params, eos, metric, metric_aux, cons, prims[, diagnostics])` | same gates | same mutation/diagnostics reference contract; translates status | `ghl_con2prim_tabulated_multi_method` |
| `nn_c2p_guess(eos, q, r, s, t)` | initialized EOS, four parsed C floats, non-null loaded model | Python float from returned `.x` | `ghl_c2p_nn_guess` |
| `nn_c2p_guess_x(...)` | identical; wrapper delegates directly to `py_nn_c2p_guess` | identical float | `ghl_c2p_nn_guess` |

`require_type` uses `PyObject_TypeCheck`, so registered type subtypes pass. C
argument parsing adds its own arity/type failures before wrapper-specific gates.
Exact upstream declarations and definitions for representative calls are in the
[bindings hub](index.md).

## EOS instance methods

`TabulatedEOS` registers two bounds methods, seven interpolation/inversion
methods, `load_nn_c2p_hdf5`, and `close`. Their initialized/function-pointer,
owned-memory, post-close, and loader gates are owned by
[tabulated EOS lifetime and loader](tabulated-eos-lifetime-and-loader.md).

## Diagnostics flow

`initialize_py_diagnostics` zeroes the embedded struct and calls pinned
`ghl_initialize_diagnostics`. `Diagnostics()` uses `tp_alloc`; the module
initializer uses `PyObject_New`; both then call that helper. Each C2P wrapper:

1. creates and initializes diagnostics when omitted, or type-checks the supplied
   wrapper;
2. passes its embedded struct and the supplied mutable `Primitive` to GRHayL;
3. decrements only a diagnostics object it created if GRHayL returns an error;
4. returns the created object directly on success, or increments and returns the
   caller-supplied object.

Fields expose whether tau/momentum/speed fixes occurred, selected routine,
backup use, and iteration count. They report upstream-written state; field
exposure does not prove a solver ran.

## Error boundary

| Failure layer | Python exception/source behavior | Exact source gate |
| --- | --- | --- |
| C argument parse, wrong wrapper, bad numeric/property deletion | parser exception or wrapper `TypeError` | `PyArg_Parse*`, `require_type`, `parse_double`, setters |
| wrong vector length or backup count | `ValueError` | `parse_vector3`, `py_initialize_params` |
| unreadable table/model path | errno-derived `OSError` with filename | `access(..., R_OK)`, `PyErr_SetFromErrnoWithFilename` |
| uninitialized/closed EOS | `RuntimeError("TabulatedEOS is not initialized.")` | `eos_ensure_initialized` |
| missing tabulated bounds function pointer | `RuntimeError` naming uninitialized GRHayL pointers | two bounds-method pointer gates |
| no loaded NN model | `GRHayLError` directing caller to `eos.load_nn_c2p_hdf5(...)` | `py_nn_c2p_guess` |
| non-success GRHayL status | `GRHayLError("<context> failed with <symbol> (<integer>)")` | `raise_ghl_error`, `ghl_error_name` |
| extension import/loader failure | stored original `ImportError`; `require_bindings` raises contextual chained `ImportError` | `src/pyghl/__init__.py` and [EOS loader](tabulated-eos-lifetime-and-loader.md) |

`ghl_error_name` maps these pinned status families before falling back to
`ghl_error_unknown`:

- dispatch/physical/C2P: `ghl_error_unknown_eos_type`,
  `ghl_error_invalid_c2p_key`, `ghl_error_neg_rho`,
  `ghl_error_neg_pressure`, `ghl_error_neg_vsq`,
  `ghl_error_c2p_max_iter`, `ghl_error_c2p_singular`,
  `ghl_error_root_not_bracketed`, `ghl_error_u0_singular`,
  `ghl_error_invalid_utsq`, `ghl_error_invalid_Z`,
  `ghl_error_newman_invalid_discriminant`;
- table bounds/interpolation: `ghl_error_table_max_rho`,
  `ghl_error_table_min_rho`, `ghl_error_table_max_ye`,
  `ghl_error_table_min_ye`, `ghl_error_table_max_T`,
  `ghl_error_table_min_T`, `ghl_error_exceed_table_vars`,
  `ghl_error_table_neg_energy`, `ghl_error_table_bisection`;
- configuration/allocation/HDF5: `ghl_error_used_disabled_hdf5`,
  `ghl_error_out_of_memory`, `ghl_error_eos_struct_is_null`,
  `ghl_error_invalid_eos_type`, `ghl_error_invalid_eos_table_type`,
  `ghl_error_could_not_open_file`,
  `ghl_error_hdf5_dataset_could_not_open`,
  `ghl_error_hdf5_dataset_could_not_read`,
  `ghl_error_hdf5_dataset_invalid_ndims`,
  `ghl_error_hdf5_dataset_size_mismatch`;
- configured bounds: `ghl_error_invalid_rho_atm`,
  `ghl_error_rho_min_gt_rho_max`, `ghl_error_invalid_press_atm`,
  `ghl_error_press_min_gt_press_max`, `ghl_error_invalid_Y_e_atm`,
  `ghl_error_Y_e_min_gt_Y_e_max`, `ghl_error_invalid_T_atm`,
  `ghl_error_T_min_gt_T_max`;
- Fermi/NN: `ghl_error_invalid_fermi_dirac_integral_key`,
  `ghl_error_nn_c2p_model_is_null`,
  `ghl_error_nn_c2p_invalid_dimensions`,
  `ghl_error_nn_c2p_invalid_input_index`,
  `ghl_error_nn_c2p_missing_array`, `ghl_error_nn_c2p_invalid_kind`,
  `ghl_error_nn_c2p_invalid_number`.

`ghl_success` is also named. Mapping a status to text does not establish which
calls can produce every status; inspect each pinned implementation.

## CPython ownership and cleanup

- EOS objects own GRHayL table/model allocations after successful
  initialization. `eos_dealloc` conditionally calls the pinned free pointer,
  clears `initialized`, then calls `Py_TYPE(self)->tp_free(self)`.
- Failed table initialization calls the available GRHayL free pointer, raises
  `GRHayLError`, and decrements the wrapper; because `initialized` is still
  false, its deallocator does not call GRHayL free. This does not make the error
  path single-owner: on an embedded-NN load failure, pinned initialization has
  already called the same free function, so the wrapper call can double-free
  table arrays. See the [EOS lifecycle owner](tabulated-eos-lifetime-and-loader.md#initialization-flow-and-defaults).
- Temporary fast-sequence references in vector/backup parsing are decremented on
  success and wrapper-detected failure paths.
- `PyModule_AddObject` is used for exception/type registration. On successful
  exception addition, the code increments `GRHayLError` to retain the global
  reference. Its exception-add failure branch decrements exception and module.
  Type-add return values are not checked in current source, so no stronger
  failure-cleanup claim is made.
- Created diagnostics are decremented on solver error; supplied diagnostics are
  not owned by the wrapper call and receive a new return reference only on
  success.

## Proof and gaps

Exact C source and parent-pin objects establish registration, parsing, cleanup,
and delegated call paths. Root tests do not directly execute wrapper property
round-trips, translated GRHayL statuses, diagnostics reference paths, or generic
construction. Upstream tests cannot prove Python ownership/error behavior.
README/examples are consumers, not assertions. See [test map](../test-map.md).

## External Ground Truth

- [Python C API: module objects](https://docs.python.org/3/c-api/module.html#c.PyModule_AddObject)
  specifies that `PyModule_AddObject` steals its object reference on success and
  requires caller cleanup on error; this bounds the registration analysis.
- [Python C API: `tp_dealloc`](https://docs.python.org/3/c-api/typeobj.html#c.PyTypeObject.tp_dealloc)
  describes destructor responsibility for owned resources and calling
  `tp_free`; this matches the EOS wrapper cleanup shape.
- [Python C API: exception handling](https://docs.python.org/3/c-api/exceptions.html#c.PyErr_Format)
  documents setting the exception indicator and returning `NULL`; wrapper
  return paths above are traced against actual project source.
