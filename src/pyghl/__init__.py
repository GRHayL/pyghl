"""Python bindings and pure-Python helpers for GRHayL."""

from __future__ import annotations

import os
from pathlib import Path

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
        original_error = str(_BINDINGS_IMPORT_ERROR)
        message = (
            "GRHayL C bindings could not be loaded. "
            f"Original loader error: {original_error}"
        )

        if "libghl" in original_error:
            grhayl_root = os.environ.get("GRHAYL_DIR")
            if grhayl_root:
                library_dir = Path(grhayl_root).expanduser().resolve() / "build" / "lib"
            else:
                checkout_library_dir = (
                    Path(__file__).resolve().parents[2]
                    / "extern"
                    / "GRHayL"
                    / "build"
                    / "lib"
                )
                library_dir = (
                    checkout_library_dir
                    if checkout_library_dir.is_dir()
                    else Path("<GRHayL-checkout>") / "build" / "lib"
                )
            message += (
                "\nFor a local or editable pyghl installation, build GRHayL and "
                "make its shared library visible before running pyghl, for example:"
                f"\n  export LD_LIBRARY_PATH={library_dir}:$LD_LIBRARY_PATH"
                "\nFor a regular installation, rebuild or reinstall pyghl so that "
                "libghl is packaged beside the extension module."
            )
        else:
            message += "\nRebuild or reinstall pyghl and inspect the loader error above."

        raise ImportError(message) from _BINDINGS_IMPORT_ERROR


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
