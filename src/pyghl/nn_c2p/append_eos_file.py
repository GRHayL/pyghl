from __future__ import annotations

import argparse
from pathlib import Path

import pyghl as ghl


def build_parser(prog: str | None = None) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=prog,
        description="Append a GRHayL nn_c2p HDF5 model into an EOS HDF5 file."
    )
    parser.add_argument("eos_hdf5", type=Path)
    parser.add_argument("nn_hdf5", nargs="?", type=Path)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Skip the stored EOS checksum verification when appending.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing embedded neural-network block.",
    )
    return parser


def main(argv: list[str] | None = None, prog: str | None = None) -> int:
    args = build_parser(prog=prog).parse_args(argv)
    existing = ghl.nn.eos_nn_metadata(args.eos_hdf5)
    if existing["contains_nn"]:
        print(
            f"Found existing neural-network data in {args.eos_hdf5} "
            f"under group {existing['group_name']}."
        )
        if not args.overwrite:
            raise SystemExit(
                f"{args.eos_hdf5} already contains {existing['group_name']!r}; "
                "rerun with --overwrite to replace it."
            )
    else:
        print(f"No embedded neural-network data found in {args.eos_hdf5}.")

    if args.nn_hdf5 is None:
        try:
            summary = ghl.nn.append_matching_installed_to_eos_file(
                args.eos_hdf5,
                overwrite=args.overwrite,
            )
        except ValueError as exc:
            raise SystemExit(
                f"{exc}\n"
                "Train a neural network for this EOS with:\n"
                f"  pyghl train {args.eos_hdf5}\n"
                "Or, if you already have a dataset:\n"
                f"  pyghl train {args.eos_hdf5} nn_training_dataset.bin"
            ) from exc
        model_desc = "the installed matching neural-network model"
    else:
        summary = ghl.nn.append_to_eos_file(
            args.eos_hdf5,
            args.nn_hdf5,
            overwrite=args.overwrite,
            require_eos_match=not args.force,
        )
        model_desc = str(args.nn_hdf5)
    action = "Overwrote" if summary["overwrite_performed"] else "Appended"
    print(f"{action} neural-network data from {model_desc} into {args.eos_hdf5}")
    print(
        "Embedded network: "
        f"hidden_layers={summary.get('n_hidden')} "
        f"hidden_dim={summary.get('hidden_dim')} "
        f"group={summary['group_name']}"
    )
    print(f"Added to EOS file at: {summary.get('embedded_utc', 'unknown')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
