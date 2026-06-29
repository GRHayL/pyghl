from __future__ import annotations

import argparse
import ast
import re
from pathlib import Path

import numpy as np

from .._nn_hdf5 import build_eos_metadata, write_nn_hdf5


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _parse_define(text: str, name: str) -> str:
    match = re.search(rf"^\s*#define\s+{re.escape(name)}\s+(.+?)\s*$", text, flags=re.MULTILINE)
    if match is None:
        raise ValueError(f"Could not find #define for {name}")
    return _strip_float_suffixes(match.group(1).strip())


def _strip_float_suffixes(text: str) -> str:
    return re.sub(r"(?<=\d)f\b", "", text)


def _parse_array(text: str, name: str):
    match = re.search(
        rf"static\s+const\s+(?:int|float)\s+{re.escape(name)}(?:\[[^\]]*\])+?\s*=\s*(\{{.*?\}});",
        text,
        flags=re.DOTALL,
    )
    if match is None:
        raise ValueError(f"Could not find array initializer for {name}")

    value_text = _strip_float_suffixes(match.group(1))
    value_text = value_text.replace("{", "[").replace("}", "]")
    return ast.literal_eval(value_text)


def _parse_audit_block(text: str) -> dict[str, str]:
    match = re.search(r"/\*\s*audit:(.*?)\*/", text, flags=re.DOTALL)
    if match is None:
        return {}

    audit: dict[str, str] = {}
    for raw_line in match.group(1).splitlines():
        line = raw_line.strip()
        if not line.startswith("*"):
            continue
        line = line[1:].strip()
        for key, value in re.findall(r"([A-Za-z0-9_]+)=([^\s]+)", line):
            audit[key] = value
    return audit


def header_to_payload(header_text: str, *, dx_eps: float) -> dict[str, object]:
    payload: dict[str, object] = {
        "in_dim": int(_parse_define(header_text, "NN_IN_DIM")),
        "hidden_dim": int(_parse_define(header_text, "NN_HIDDEN_DIM")),
        "n_hidden": int(_parse_define(header_text, "NN_N_HIDDEN")),
        "out_dim": int(_parse_define(header_text, "NN_OUT_DIM")),
        "x_eps": np.float32(float(_parse_define(header_text, "NN_X_EPS"))),
        "y_eps": np.float32(float(_parse_define(header_text, "NN_Y_EPS"))),
        "q_idx": int(_parse_define(header_text, "NN_Q_IDX")),
        "s_idx": int(_parse_define(header_text, "NN_S_IDX")),
        "dx_eps": np.float32(dx_eps),
        "x_kind": np.asarray(_parse_array(header_text, "nn_x_kind"), dtype=np.int32),
        "x_lo": np.asarray(_parse_array(header_text, "nn_x_lo"), dtype=np.float32),
        "x_hi": np.asarray(_parse_array(header_text, "nn_x_hi"), dtype=np.float32),
        "x_invrng": np.asarray(_parse_array(header_text, "nn_x_invrng"), dtype=np.float32),
        "out_kind": np.asarray(_parse_array(header_text, "nn_out_kind"), dtype=np.int32),
        "out_lo": np.asarray(_parse_array(header_text, "nn_out_lo"), dtype=np.float32),
        "out_hi": np.asarray(_parse_array(header_text, "nn_out_hi"), dtype=np.float32),
        "out_invrng": np.asarray(_parse_array(header_text, "nn_out_invrng"), dtype=np.float32),
        "W_in": np.asarray(_parse_array(header_text, "nn_W_in"), dtype=np.float32),
        "b_in": np.asarray(_parse_array(header_text, "nn_b_in"), dtype=np.float32),
        "W_out": np.asarray(_parse_array(header_text, "nn_W_out"), dtype=np.float32),
        "b_out": np.asarray(_parse_array(header_text, "nn_b_out"), dtype=np.float32),
        "audit": _parse_audit_block(header_text),
    }

    n_hidden = int(payload["n_hidden"])
    hidden_dim = int(payload["hidden_dim"])
    if n_hidden > 1:
        payload["W_hid"] = np.asarray(_parse_array(header_text, "nn_W_hid"), dtype=np.float32)
        payload["b_hid"] = np.asarray(_parse_array(header_text, "nn_b_hid"), dtype=np.float32)
    else:
        payload["W_hid"] = np.zeros((0, hidden_dim, hidden_dim), dtype=np.float32)
        payload["b_hid"] = np.zeros((0, hidden_dim), dtype=np.float32)

    return payload


def write_hdf5(payload: dict[str, object], output_path: Path, *, source_header: Path) -> None:
    mapped = {
        "dims": {
            "in_dim": np.int32(payload["in_dim"]),
            "hidden_dim": np.int32(payload["hidden_dim"]),
            "n_hidden": np.int32(payload["n_hidden"]),
            "out_dim": np.int32(payload["out_dim"]),
        },
        "meta": {
            "q_idx": np.int32(payload["q_idx"]),
            "s_idx": np.int32(payload["s_idx"]),
            "y_eps": np.float32(payload["y_eps"]),
            "dx_eps": np.float32(payload["dx_eps"]),
        },
        "scaling": {
            "x_eps": np.float32(payload["x_eps"]),
            "x_kind": np.asarray(payload["x_kind"], dtype=np.int32),
            "x_lo": np.asarray(payload["x_lo"], dtype=np.float32),
            "x_hi": np.asarray(payload["x_hi"], dtype=np.float32),
            "x_invrng": np.asarray(payload["x_invrng"], dtype=np.float32),
            "out_kind": np.asarray(payload["out_kind"], dtype=np.int32),
            "out_lo": np.asarray(payload["out_lo"], dtype=np.float32),
            "out_hi": np.asarray(payload["out_hi"], dtype=np.float32),
            "out_invrng": np.asarray(payload["out_invrng"], dtype=np.float32),
        },
        "layers": {
            "W_in": np.asarray(payload["W_in"], dtype=np.float32),
            "b_in": np.asarray(payload["b_in"], dtype=np.float32),
            "W_hid": np.asarray(payload["W_hid"], dtype=np.float32),
            "b_hid": np.asarray(payload["b_hid"], dtype=np.float32),
            "W_out": np.asarray(payload["W_out"], dtype=np.float32),
            "b_out": np.asarray(payload["b_out"], dtype=np.float32),
        },
        "audit": dict(payload["audit"]),
    }
    write_nn_hdf5(mapped, output_path, root_attrs={"source_header": str(source_header)})


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert a GRHayL nn_c2p weights header into an HDF5 model file."
    )
    parser.add_argument("header", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--dx-eps", type=float, default=1.0e-12)
    parser.add_argument("--eos-file", type=Path)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    header_text = _read_text(args.header)
    payload = header_to_payload(header_text, dx_eps=args.dx_eps)
    if args.eos_file is not None:
        payload["source_eos"] = build_eos_metadata(args.eos_file)
    write_hdf5(payload, args.output, source_header=args.header)
    print(f"Wrote HDF5 model: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
