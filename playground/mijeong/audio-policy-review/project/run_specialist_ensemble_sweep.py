from __future__ import annotations

import argparse
import subprocess
import sys

import config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--assignment",
        choices=["predefined", "automatic"],
        required=True,
    )
    args = parser.parse_args()

    for fraction in config.SPECIALIST_NON_EXPERT_FRACTIONS:
        command = [
            sys.executable,
            "ensemble_specialists.py",
            "--assignment",
            args.assignment,
            "--ratio",
            str(float(fraction)),
        ]
        print("+", " ".join(command), flush=True)
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
