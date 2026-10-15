from __future__ import annotations

import subprocess
import sys

import config


COMMANDS = ['train_yamnet_predefined.py']


def main() -> None:
    for fraction in config.SPECIALIST_NON_EXPERT_FRACTIONS:
        ratio = str(float(fraction))
        for script in COMMANDS:
            command = [sys.executable, script, "--ratio", ratio]
            print("+", " ".join(command), flush=True)
            subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
