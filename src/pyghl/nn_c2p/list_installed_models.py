from __future__ import annotations

import argparse

import pyghl as ghl


def build_parser(prog: str | None = None) -> argparse.ArgumentParser:
    return argparse.ArgumentParser(
        prog=prog,
        description="List packaged GRHayL nn_c2p models that can be auto-matched to EOS files.",
    )


def main(argv: list[str] | None = None, prog: str | None = None) -> int:
    build_parser(prog=prog).parse_args(argv)
    models = ghl.nn.installed_nn_models()
    if not models:
        print("No installed neural-network EOS models were found.")
        return 0

    print("Installed neural-network EOS models:")
    for model in models:
        print(
            f"- {model['model_filename']}: "
            f"source_eos={model['source_eos_filename'] or 'unknown'} "
            f"hidden_layers={model['n_hidden']} "
            f"hidden_dim={model['hidden_dim']} "
            f"canonical_md5={model['canonical_md5']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
