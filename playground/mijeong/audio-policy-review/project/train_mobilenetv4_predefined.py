from __future__ import annotations

import argparse
import sys

import config
from train_mobilenetv4 import main as base_main


def main() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--ratio",
        type=float,
        required=True,
    )
    known, remaining = parser.parse_known_args()

    csv_path = config.predefined_csv("mobilenetv4", known.ratio)
    output_path = config.predefined_output("mobilenetv4", known.ratio)

    sys.argv = [sys.argv[0]] + remaining
    if "--csv" not in sys.argv:
        sys.argv += ["--csv", str(csv_path)]
    if "--output" not in sys.argv:
        sys.argv += ["--output", str(output_path)]

    base_main()


if __name__ == "__main__":
    main()
