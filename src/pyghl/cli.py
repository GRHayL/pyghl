from __future__ import annotations

import argparse
from collections.abc import Sequence
from importlib.metadata import PackageNotFoundError, version
import sys


def _package_version() -> str:
    try:
        return version("pyghl")
    except PackageNotFoundError:
        return "unknown"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="pyghl")
    parser.add_argument(
        "-v",
        "--version",
        action="store_true",
        help="Show version and exit.",
    )
    subparsers = parser.add_subparsers(dest="command", metavar="command")

    train = subparsers.add_parser(
        "train",
        help="Train a neural-network con2prim model for an EOS table.",
        add_help=False,
    )
    train.add_argument("args", nargs=argparse.REMAINDER)

    append_eos = subparsers.add_parser(
        "append-eos",
        help="Append a trained neural-network model to an EOS HDF5 file.",
        add_help=False,
    )
    append_eos.add_argument("args", nargs=argparse.REMAINDER)

    append = subparsers.add_parser(
        "append",
        help="Alias for append-eos.",
        add_help=False,
    )
    append.add_argument("args", nargs=argparse.REMAINDER)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = build_parser()

    if not argv or argv[0] in ("-h", "--help"):
        parser.print_help()
        return 0 if argv else 2

    if argv[0] in ("-v", "--version"):
        print(f"pyghl {_package_version()}")
        return 0

    command = argv[0]
    command_args = argv[1:]

    if command == "train":
        from .nn_c2p.nn_c2p_train import main as train_main

        return train_main(command_args, prog="pyghl train")

    if command in ("append-eos", "append"):
        from .nn_c2p.append_eos_file import main as append_main

        return append_main(command_args, prog=f"pyghl {command}")

    parser.error(f"unknown command: {command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
