from __future__ import annotations

import struct
from pathlib import Path

import h5py
import numpy as np


HEADER_SIZE_BYTES = 24
HEADER_FORMAT = "<QQQ"
SUPPORTED_FLOAT_BITS = (32, 64)
INPUT_COLUMN_INDICES = (11, 12, 13, 14)
TARGET_COLUMN_INDICES_RHO_T_W = (0, 1, 3)
TARGET_COLUMN_INDICES_X = (15,)

SOLVER_REGION_MODES = ("solver_region", "solver_soft_bins", "solver_gate")

def read_training_dataset(
    filename: str | Path,
    *,
    target_mode: str = "x_correction",
) -> np.ndarray:
    if target_mode in SOLVER_REGION_MODES:
        raise ValueError(
            f"target_mode {target_mode!r} requires read_solver_label_dataset(), "
            "not the legacy binary dataset reader."
        )
    file_path = Path(filename)
    print(f"Reading training dataset from {str(file_path)!r}")

    with file_path.open("rb") as f:
        header = f.read(HEADER_SIZE_BYTES)
        if len(header) != HEADER_SIZE_BYTES:
            raise ValueError(
                f"Incomplete header in {str(file_path)!r}: expected {HEADER_SIZE_BYTES} bytes (3x uint64), got {len(header)}."
            )

        float_size_bytes, n_floats_per_block, n_blocks = struct.unpack(
            HEADER_FORMAT, header
        )

        float_size_bits = 8 * float_size_bytes
        if float_size_bits not in SUPPORTED_FLOAT_BITS:
            raise ValueError(
                f"Invalid float size in header of {str(file_path)!r}: {float_size_bits} bits "
                f"(expected {SUPPORTED_FLOAT_BITS[0]} or {SUPPORTED_FLOAT_BITS[1]})."
            )
        print(f"Input data floating point size: {float_size_bits} bits")

        le_dtype = np.dtype("<f4" if float_size_bits == 32 else "<f8")
        expected_count = int(n_floats_per_block) * int(n_blocks)
        data = f.read()

    itemsize = le_dtype.itemsize
    if len(data) % itemsize != 0:
        raise ValueError(
            f"Incomplete float data in {str(file_path)!r}: data section size {len(data)} bytes is not a multiple of "
            f"float size {itemsize} bytes."
        )

    got_count = len(data) // itemsize
    if got_count != expected_count:
        raise ValueError(
            f"Float count mismatch in {str(file_path)!r}: expected {expected_count} floats "
            f"(n_blocks={n_blocks} * n_floats_per_block={n_floats_per_block}), got {got_count}."
        )

    arr = np.frombuffer(data, dtype=le_dtype, count=expected_count).reshape(
        (int(n_blocks), int(n_floats_per_block))
    )
    if target_mode == "rho_T_W":
        target_columns = TARGET_COLUMN_INDICES_RHO_T_W
    elif target_mode in ("x_correction", "x_best_correction"):
        target_columns = TARGET_COLUMN_INDICES_X
    else:
        raise ValueError(
            f"Unsupported target_mode {target_mode!r}. "
            "Expected 'x_correction', 'x_best_correction', or 'rho_T_W'."
        )
    required_columns = max(INPUT_COLUMN_INDICES + target_columns) + 1
    if arr.shape[1] < required_columns:
        raise ValueError(
            f"Expected at least {required_columns} columns per block, got {arr.shape[1]}."
        )

    print(f"Successfully read {n_blocks} training rows from {str(file_path)!r}")
    columns = INPUT_COLUMN_INDICES + target_columns
    return arr[:, columns].astype(np.float32, copy=False)


def _require_dataset(h5f: h5py.File, name: str):
    if name not in h5f:
        raise KeyError(f"Missing dataset {name!r} in solver label file.")
    return h5f[name]


def _read_array(h5f: h5py.File, name: str, *, dtype=None) -> np.ndarray:
    arr = np.asarray(_require_dataset(h5f, name)[()])
    if dtype is not None:
        arr = arr.astype(dtype, copy=False)
    return arr


def read_solver_label_dataset(
    filename: str | Path,
    *,
    target_mode: str,
) -> dict[str, np.ndarray]:
    if target_mode not in SOLVER_REGION_MODES:
        raise ValueError(
            f"Unsupported solver target_mode {target_mode!r}. "
            f"Expected one of {SOLVER_REGION_MODES!r}."
        )

    file_path = Path(filename)
    print(f"Reading solver label dataset from {str(file_path)!r}")
    with h5py.File(file_path, "r") as h5f:
        X = np.stack(
            (
                _read_array(h5f, "states/q", dtype=np.float32),
                _read_array(h5f, "states/r", dtype=np.float32),
                _read_array(h5f, "states/s", dtype=np.float32),
                _read_array(h5f, "states/t", dtype=np.float32),
            ),
            axis=1,
        )
        y_grid = _read_array(h5f, "grid/y_uniform", dtype=np.float32)
        if "labels/state_type" in h5f:
            state_kind = _read_array(h5f, "labels/state_type", dtype=np.int32)
        else:
            state_kind = _read_array(h5f, "labels/state_kind", dtype=np.int32)
        sample_weight = _read_array(h5f, "labels/sample_weight", dtype=np.float32)

        out: dict[str, np.ndarray] = {
            "X": X,
            "y_grid": y_grid,
            "state_kind": state_kind,
            "state_type": state_kind,
            "sample_weight": sample_weight,
            "x_lo": _read_array(h5f, "states/x_lo", dtype=np.float32),
            "x_hi": _read_array(h5f, "states/x_hi", dtype=np.float32),
            "x_exact": _read_array(h5f, "states/x_exact", dtype=np.float32),
            "y_exact": _read_array(h5f, "states/y_exact", dtype=np.float32),
            "iter_min_success": _read_array(
                h5f, "labels/iter_min_success", dtype=np.float32
            ),
            "best_baseline_id": _read_array(
                h5f, "labels/best_baseline_id", dtype=np.int32
            ),
            "best_baseline_cost": _read_array(
                h5f, "labels/best_baseline_cost", dtype=np.float32
            ),
            "best_simple_baseline_id": _read_array(
                h5f, "labels/best_simple_baseline_id", dtype=np.int32
            ),
            "phase_a_beats_simple_baseline": _read_array(
                h5f, "labels/phase_a_beats_simple_baseline", dtype=np.int8
            ),
            "recommended_id": _read_array(
                h5f, "labels/recommended_id", dtype=np.int32
            ),
        }

        if target_mode == "solver_region":
            out["y_fast_mask"] = _read_array(
                h5f, "labels/y_fast_mask", dtype=np.bool_
            )
            out["y_robust_mask"] = _read_array(
                h5f, "labels/y_robust_mask", dtype=np.bool_
            )
        elif target_mode == "solver_soft_bins":
            out["soft_cost"] = _read_array(
                h5f, "labels/soft_cost", dtype=np.float32
            )
            out["soft_prob"] = _read_array(
                h5f, "labels/soft_prob", dtype=np.float32
            )
        elif target_mode == "solver_gate":
            out["gate_target"] = _read_array(
                h5f, "labels/gate_oracle_use_solver_region", dtype=np.float32
            )

    if X.ndim != 2 or X.shape[1] != 4:
        raise ValueError(
            f"Expected solver features X to have shape (N,4). Got {tuple(X.shape)}."
        )
    if y_grid.ndim != 1 or y_grid.shape[0] < 2:
        raise ValueError(
            f"Expected y_grid to have shape (K,) with K>=2. Got {tuple(y_grid.shape)}."
        )
    if state_kind.shape[0] != X.shape[0]:
        raise ValueError(
            f"state_kind length {state_kind.shape[0]} does not match N={X.shape[0]}."
        )
    if sample_weight.shape[0] != X.shape[0]:
        raise ValueError(
            f"sample_weight length {sample_weight.shape[0]} does not match N={X.shape[0]}."
        )
    print(f"Successfully read {X.shape[0]} solver-label rows from {str(file_path)!r}")
    return out


def read_solver_cost_label_dataset(
    cost_map_h5: str | Path,
    label_h5: str | Path,
    *,
    target_mode: str,
) -> dict[str, np.ndarray]:
    # Current implementation stores the training-facing features inside the
    # label HDF5 for simpler state-level splits, but keep the two-file API from
    # implementation.md so callers can follow that contract directly.
    _ = Path(cost_map_h5)
    return read_solver_label_dataset(label_h5, target_mode=target_mode)
