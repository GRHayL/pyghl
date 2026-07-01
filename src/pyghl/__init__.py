"""Python bindings and pure-Python helpers for GRHayL."""

from __future__ import annotations

_BINDINGS_IMPORT_ERROR: ImportError | None = None
_BINDINGS_AVAILABLE = False

try:
    from ._pyghl import (
        ADMAux,
        C2P_FONT1D,
        C2P_NEWMAN1D,
        C2P_NEWMAN1D_ENTROPY,
        C2P_NOBLE1D,
        C2P_NOBLE1D_ENTROPY,
        C2P_NOBLE1D_ENTROPY2,
        C2P_NOBLE2D,
        C2P_NONE,
        C2P_PALENZUELA1D,
        C2P_PALENZUELA1D_ENTROPY,
        Conservative,
        Diagnostics,
        GRHayLError,
        Metric,
        Params,
        Primitive,
        TabulatedEOS,
        compute_ADM_auxiliaries,
        compute_conservs,
        compute_SU_Bsq_Ssq_BdotS,
        guess_primitives,
        initialize_diagnostics,
        initialize_metric,
        initialize_params,
        initialize_primitives,
        limit_utilde_and_compute_v,
        limit_v_and_compute_u0,
        nn_c2p_guess,
        nn_c2p_guess_x,
        tabulated_con2prim_multi_method,
        tabulated_Palenzuela1D_energy,
        undensitize_conservatives,
    )
except ImportError as exc:
    _BINDINGS_IMPORT_ERROR = exc
else:
    _BINDINGS_AVAILABLE = True
    from . import eos

from . import nn


def require_bindings() -> None:
    """Raise the original extension import error if C bindings are unavailable."""
    if _BINDINGS_IMPORT_ERROR is not None:
        raise ImportError(
            "GRHayL C bindings are unavailable. Rebuild/reinstall the Python "
            "package or fix LD_LIBRARY_PATH."
        ) from _BINDINGS_IMPORT_ERROR


__all__ = [
    "_BINDINGS_AVAILABLE",
    "nn",
    "require_bindings",
]

if _BINDINGS_AVAILABLE:
    __all__ += [
        "ADMAux",
        "C2P_FONT1D",
        "C2P_NEWMAN1D",
        "C2P_NEWMAN1D_ENTROPY",
        "C2P_NOBLE1D",
        "C2P_NOBLE1D_ENTROPY",
        "C2P_NOBLE1D_ENTROPY2",
        "C2P_NOBLE2D",
        "C2P_NONE",
        "C2P_PALENZUELA1D",
        "C2P_PALENZUELA1D_ENTROPY",
        "Conservative",
        "Diagnostics",
        "GRHayLError",
        "Metric",
        "Params",
        "Primitive",
        "TabulatedEOS",
        "compute_ADM_auxiliaries",
        "compute_conservs",
        "compute_SU_Bsq_Ssq_BdotS",
        "eos",
        "guess_primitives",
        "initialize_diagnostics",
        "initialize_metric",
        "initialize_params",
        "initialize_primitives",
        "limit_utilde_and_compute_v",
        "limit_v_and_compute_u0",
        "nn_c2p_guess",
        "nn_c2p_guess_x",
        "tabulated_con2prim_multi_method",
        "tabulated_Palenzuela1D_energy",
        "undensitize_conservatives",
    ]
