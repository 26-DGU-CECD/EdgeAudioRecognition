from __future__ import annotations

import argparse
import sys

import config
from train_yamnet import main as base_main


def main() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--ratio",
        type=float,
        required=True,
    )
    known, remaining = parser.parse_known_args()

    csv_path = config.auto_csv("yamnet", known.ratio)
    output_path = config.auto_output("yamnet", known.ratio)

    sys.argv = [sys.argv[0]] + remaining
    if "--csv" not in sys.argv:
        sys.argv += ["--csv", str(csv_path)]
    if "--output" not in sys.argv:
        sys.argv += ["--output", str(output_path)]
    if "--reuse-cache-dir" not in sys.argv:
        sys.argv += [
            "--reuse-cache-dir",
            str(config.YAMNET_OUTPUT / "embedding_cache"),
        ]
    base_main()


if __name__ == "__main__":
    main()
