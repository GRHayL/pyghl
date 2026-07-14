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

    append = subparsers.add_parser(
        "append",
        help="Append a trained neural-network model to an EOS HDF5 file.",
        add_help=False,
    )
    append.add_argument("args", nargs=argparse.REMAINDER)

    check_eos = subparsers.add_parser(
        "check-eos",
        help="Inspect an EOS HDF5 file for embedded neural-network data.",
        add_help=False,
    )
    check_eos.add_argument("args", nargs=argparse.REMAINDER)

    list_models = subparsers.add_parser(
        "list-models",
        help="List installed neural-network models.",
        add_help=False,
    )
    list_models.add_argument("args", nargs=argparse.REMAINDER)

    remove_eos_nn = subparsers.add_parser(
        "remove-eos-nn",
        help="Remove embedded neural-network data from an EOS HDF5 file.",
        add_help=False,
    )
    remove_eos_nn.add_argument("args", nargs=argparse.REMAINDER)

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
        from . import require_bindings

        try:
            require_bindings()
        except ImportError as exc:
            parser.exit(1, f"pyghl train: error: {exc}\n")
        from .nn_c2p.nn_c2p_train import main as train_main

        return train_main(command_args, prog="pyghl train")

    if command in ("append-eos", "append"):
        from .nn_c2p.append_eos_file import main as append_main

        return append_main(command_args, prog=f"pyghl {command}")

    if command == "check-eos":
        from .nn_c2p.check_eos import main as check_main

        return check_main(command_args, prog="pyghl check-eos")

    if command == "list-models":
        from .nn_c2p.list_installed_models import main as list_main

        return list_main(command_args, prog="pyghl list-models")

    if command == "remove-eos-nn":
        from .nn_c2p.remove_eos_nn import main as remove_main

        return remove_main(command_args, prog="pyghl remove-eos-nn")

    parser.error(f"unknown command: {command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
