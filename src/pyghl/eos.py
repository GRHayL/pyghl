"""EOS wrappers for GRHayL.

This module currently exposes tabulated EOS bindings first.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Union

from . import _pyghl

PathLike = Union[str, os.PathLike[str]]
TabulatedEOS = _pyghl.TabulatedEOS


def _resolve_table_path(table: PathLike) -> str:
    path = Path(table).expanduser()
    if path.is_file():
        return str(path.resolve())

    if path.parent != Path("."):
        raise FileNotFoundError(f"EOS table not found: {path}")

    table_name = path.name
    env_dir = os.environ.get("GRHAYL_EOS_TABLE_DIR")
    if env_dir:
        base = Path(env_dir).expanduser()
        for suffix in ("", ".h5", ".hdf5"):
            candidate = base / f"{table_name}{suffix}"
            if candidate.is_file():
                return str(candidate.resolve())

    raise FileNotFoundError(
        f"Could not resolve EOS table '{table}'. Pass a full path or set "
        "GRHAYL_EOS_TABLE_DIR."
    )


def initialize_tabulated_eos_functions_and_params(
    params: _pyghl.Params,
    table: PathLike,
    **kwargs: float,
) -> TabulatedEOS:
    """Initialize tabulated EOS functions and parameters.

    Parameters mirror GRHayL's tabulated initializer. By default this uses
    conservative values similar to the GRHayL tabulated unit tests.
    """
    table_path = _resolve_table_path(table)
    return _pyghl.initialize_tabulated_eos_functions_and_params(
        params,
        table_path,
        **kwargs,
    )
