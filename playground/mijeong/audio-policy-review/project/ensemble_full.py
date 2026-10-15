from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import config
from common_metadata import read_class_names
from evaluation_utils import (
    align_prediction_files,
    save_classification_artifacts,
    stable_softmax,
)


def main() -> None:
    class_names = read_class_names(config.CLASS_NAMES_PATH)
    files = {
        "efficientat": config.EFFICIENTAT_OUTPUT / "test_predictions.npz",
        "yamnet": config.YAMNET_OUTPUT / "test_predictions.npz",
        "mobilenetv4": config.MOBILENETV4_OUTPUT / "test_predictions.npz",
    }
    paths, y_true, _, logits_by_model = align_prediction_files(files)

    probabilities = {
        name: stable_softmax(logits)
        for name, logits in logits_by_model.items()
    }
    ensemble_probability = np.mean(
        np.stack(list(probabilities.values()), axis=0),
        axis=0,
    )
    ensemble_probability /= np.clip(
        ensemble_probability.sum(axis=1, keepdims=True),
        1e-12,
        None,
    )

    output = Path(config.FULL_ENSEMBLE_OUTPUT)
    output.mkdir(parents=True, exist_ok=True)
    (output / "ensemble_members.json").write_text(
        json.dumps(
            {
                "method": "equal soft voting",
                "members": list(files),
                "weights": {name: 1.0 / len(files) for name in files},
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    save_classification_artifacts(
        output,
        "test",
        y_true,
        np.log(np.clip(ensemble_probability, 1e-12, 1.0)),
        paths,
        class_names,
        extra_arrays={"probabilities": ensemble_probability},
    )


if __name__ == "__main__":
    main()
