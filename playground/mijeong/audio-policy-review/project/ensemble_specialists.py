from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

import config
from common_metadata import read_class_names
from evaluation_utils import (
    align_prediction_files,
    per_class_f1,
    save_classification_artifacts,
    stable_softmax,
)


MODEL_NAMES = list(config.MODEL_NAMES)


def resolve_paths(assignment: str, fraction: float):
    if assignment == "predefined":
        output_fn = config.predefined_output
        metadata_dir = config.predefined_metadata_dir(fraction)
        ensemble_output = config.predefined_ensemble_output(fraction)
    elif assignment == "automatic":
        output_fn = config.auto_output
        metadata_dir = config.auto_metadata_dir(fraction)
        ensemble_output = config.auto_ensemble_output(fraction)
    else:
        raise ValueError(assignment)

    val_files = {
        model: output_fn(model, fraction) / "val_predictions.npz"
        for model in MODEL_NAMES
    }
    test_files = {
        model: output_fn(model, fraction) / "test_predictions.npz"
        for model in MODEL_NAMES
    }
    return val_files, test_files, metadata_dir, ensemble_output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--assignment",
        choices=["predefined", "automatic"],
        required=True,
    )
    parser.add_argument("--ratio", type=float, required=True)
    args = parser.parse_args()

    fraction = float(args.ratio)
    class_names = read_class_names(config.CLASS_NAMES_PATH)
    (
        val_files,
        test_files,
        metadata_dir,
        output,
    ) = resolve_paths(args.assignment, fraction)

    _, val_true, _, val_logits = align_prediction_files(val_files)
    test_paths, test_true, _, test_logits = align_prediction_files(
        test_files
    )

    assignment_data = json.loads(
        (metadata_dir / "specialist_assignment.json").read_text(
            encoding="utf-8"
        )
    )
    experts = assignment_data["expert_classes"]

    raw_weights = np.zeros(
        (len(MODEL_NAMES), len(class_names)),
        dtype=np.float64,
    )
    validation_f1 = {}

    for model_index, model_name in enumerate(MODEL_NAMES):
        class_f1 = per_class_f1(
            val_true,
            val_logits[model_name].argmax(axis=1),
            len(class_names),
        )
        validation_f1[model_name] = class_f1
        weights = np.maximum(
            class_f1,
            float(config.SPECIALIST_WEIGHT_FLOOR),
        ) ** float(config.SPECIALIST_WEIGHT_POWER)

        for class_name in experts[model_name]:
            weights[class_names.index(class_name)] *= float(
                config.SPECIALIST_EXPERT_WEIGHT_BOOST
            )
        raw_weights[model_index] = weights

    normalized_weights = raw_weights / np.clip(
        raw_weights.sum(axis=0, keepdims=True),
        1e-12,
        None,
    )

    probabilities = np.stack(
        [
            stable_softmax(test_logits[model_name])
            for model_name in MODEL_NAMES
        ],
        axis=0,
    )
    combined = (
        probabilities * normalized_weights[:, None, :]
    ).sum(axis=0)
    combined /= np.clip(
        combined.sum(axis=1, keepdims=True),
        1e-12,
        None,
    )

    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        normalized_weights.T,
        index=class_names,
        columns=MODEL_NAMES,
    ).to_csv(output / "classwise_model_weights.csv")

    pd.DataFrame(
        validation_f1,
        index=class_names,
    ).to_csv(
        output / "specialist_validation_per_class_f1.csv"
    )

    (output / "weighting_config.json").write_text(
        json.dumps(
            {
                "assignment": args.assignment,
                "ratio_tag": config.ratio_tag(fraction),
                "nonexpert_fraction": fraction,
                "basis": "specialist validation per-class F1",
                "floor": config.SPECIALIST_WEIGHT_FLOOR,
                "power": config.SPECIALIST_WEIGHT_POWER,
                "expert_boost": config.SPECIALIST_EXPERT_WEIGHT_BOOST,
                "expert_classes": experts,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    save_classification_artifacts(
        output,
        "test",
        test_true,
        np.log(np.clip(combined, 1e-12, 1.0)),
        test_paths,
        class_names,
        extra_arrays={
            "probabilities": combined,
            "classwise_weights": normalized_weights,
        },
    )


if __name__ == "__main__":
    main()
