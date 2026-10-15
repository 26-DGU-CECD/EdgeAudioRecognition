from __future__ import annotations

import sys
import config
from train_yamnet import main


if __name__ == "__main__":
    if "--csv" not in sys.argv:
        sys.argv += ["--csv", str(config.YAMNET_SPECIALIST_CSV)]
    if "--output" not in sys.argv:
        sys.argv += ["--output", str(config.YAMNET_SPECIALIST_OUTPUT)]
    if "--reuse-cache-dir" not in sys.argv:
        sys.argv += [
            "--reuse-cache-dir",
            str(config.YAMNET_OUTPUT / "embedding_cache"),
        ]
    main()
