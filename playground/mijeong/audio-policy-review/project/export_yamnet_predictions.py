from __future__ import annotations

from pathlib import Path

import numpy as np
import tensorflow as tf

import config
from common_metadata import read_class_names
from evaluation_utils import save_classification_artifacts


def main() -> None:
    class_names = read_class_names(config.CLASS_NAMES_PATH)
    output = Path(config.YAMNET_OUTPUT)
    model = tf.keras.models.load_model(output / "best.keras")

    for split in ["val", "test"]:
        cache = np.load(
            output / "embedding_cache" / f"{split}.npz",
            allow_pickle=False,
        )
        logits = model.predict(
            cache["x"],
            batch_size=config.BATCH_SIZE,
            verbose=0,
        )
        save_classification_artifacts(
            output,
            split,
            cache["y"],
            logits,
            cache["paths"],
            class_names,
        )


if __name__ == "__main__":
    main()
