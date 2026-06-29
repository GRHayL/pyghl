from __future__ import annotations

import argparse
from pathlib import Path

import pyghl as ghl


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Inspect an EOS HDF5 file for embedded GRHayL nn_c2p data."
    )
    parser.add_argument("eos_hdf5", type=Path)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    info = ghl.nn.eos_nn_metadata(args.eos_hdf5)
    print(f"EOS file: {info['eos_filename']}")
    print(f"Path: {info['eos_path']}")
    print(f"Embedded group: {info['group_name']}")
    if not info["contains_nn"]:
        print("Embedded neural-network data: not found")
        return 0

    print("Embedded neural-network data: found")
    print(f"Format: {info.get('format')} v{info.get('format_version')}")
    print(
        "Network shape: "
        f"in={info.get('in_dim')} hidden={info.get('hidden_dim')} "
        f"layers={info.get('n_hidden')} out={info.get('out_dim')}"
    )
    print(f"Standalone model file: {info.get('nn_hdf5_filename')}")
    print(f"Added to EOS file: {info.get('embedded_utc', 'unknown')}")
    print(f"EOS hash kind: {info.get('eos_hash_kind')}")
    if "source_model_eos_filename" in info:
        print(f"Model source EOS: {info['source_model_eos_filename']}")
    if "canonical_md5" in info:
        print(f"Current EOS canonical md5: {info['canonical_md5']}")
    if "source_model_eos_md5" in info:
        print(f"Model source EOS md5: {info['source_model_eos_md5']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
