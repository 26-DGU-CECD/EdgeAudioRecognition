from __future__ import annotations

import sys
import config
from train_efficientat import main


if __name__ == "__main__":
    if "--csv" not in sys.argv:
        sys.argv += ["--csv", str(config.EFFICIENTAT_SPECIALIST_CSV)]
    if "--output" not in sys.argv:
        sys.argv += ["--output", str(config.EFFICIENTAT_SPECIALIST_OUTPUT)]
    main()
