from __future__ import annotations

import argparse
from pathlib import Path

import pyghl as ghl


def build_parser(prog: str | None = None) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=prog,
        description="Remove the embedded GRHayL nn_c2p block from an EOS HDF5 file."
    )
    parser.add_argument("eos_hdf5", type=Path)
    parser.add_argument("--verbose", action="store_true")
    return parser


def main(argv: list[str] | None = None, prog: str | None = None) -> int:
    args = build_parser(prog=prog).parse_args(argv)
    info = ghl.nn.eos_nn_metadata(args.eos_hdf5)
    if not info["contains_nn"]:
        print(f"No embedded neural-network dataset found in {args.eos_hdf5}; nothing to do.")
        return 0

    summary = ghl.nn.remove_from_eos_file(args.eos_hdf5)
    print(f"Removed embedded neural-network data from {summary['eos_filename']}.")
    if args.verbose:
        print(f"Removed group: {summary['group_name']}")
        if summary["removed_nn_hdf5_filename"] is not None:
            print(f"Removed model payload: {summary['removed_nn_hdf5_filename']}")
        print(f"Raw file md5 before: {summary['raw_md5_before']}")
        print(f"Raw file md5 after : {summary['raw_md5_after']}")
        print(f"Canonical EOS md5 before: {summary['canonical_md5_before']}")
        print(f"Canonical EOS md5 after : {summary['canonical_md5_after']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
