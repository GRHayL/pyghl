from __future__ import annotations

import struct
from pathlib import Path

import numpy as np


HEADER_SIZE_BYTES = 24
HEADER_FORMAT = "<QQQ"
SUPPORTED_FLOAT_BITS = (32, 64)
INPUT_COLUMN_INDICES = (11, 12, 13, 14)
TARGET_COLUMN_INDICES_X = (15,)

def read_training_dataset(
    filename: str | Path,
    *,
    target_mode: str = "x_correction",
) -> np.ndarray:
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
    if target_mode in ("x_correction", "x_best_correction"):
        target_columns = TARGET_COLUMN_INDICES_X
    else:
        raise ValueError(
            f"Unsupported target_mode {target_mode!r}. "
            "Expected 'x_correction' or 'x_best_correction'."
        )
    required_columns = max(INPUT_COLUMN_INDICES + target_columns) + 1
    if arr.shape[1] < required_columns:
        raise ValueError(
            f"Expected at least {required_columns} columns per block, got {arr.shape[1]}."
        )

    print(f"Successfully read {n_blocks} training rows from {str(file_path)!r}")
    columns = INPUT_COLUMN_INDICES + target_columns
    return arr[:, columns].astype(np.float32, copy=False)
