from __future__ import annotations

import sys
import config
from train_mobilenetv4 import main


if __name__ == "__main__":
    if "--csv" not in sys.argv:
        sys.argv += ["--csv", str(config.MOBILENETV4_SPECIALIST_CSV)]
    if "--output" not in sys.argv:
        sys.argv += ["--output", str(config.MOBILENETV4_SPECIALIST_OUTPUT)]
    main()
