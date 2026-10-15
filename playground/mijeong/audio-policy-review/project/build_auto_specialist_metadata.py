from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

import config
from common_metadata import read_class_names
from evaluation_utils import align_prediction_files, per_class_f1
from specialist_data_utils import build_specialist_dataframe


MODEL_NAMES = list(config.MODEL_NAMES)


def automatic_assignment(
    f1_table: pd.DataFrame,
    class_names: list[str],
) -> dict[str, list[str]]:
    foreground = [name for name in class_names if name != "background"]

    owner = {
        class_name: str(f1_table.loc[class_name].idxmax())
        for class_name in foreground
    }

    minimum = int(config.AUTO_MIN_EXPERT_CLASSES_PER_MODEL)
    while True:
        counts = {
            model: sum(value == model for value in owner.values())
            for model in MODEL_NAMES
        }
        target = next(
            (model for model in MODEL_NAMES if counts[model] < minimum),
            None,
        )
        if target is None:
            break

        candidates = []
        for class_name, current in owner.items():
            if current == target or counts[current] <= minimum:
                continue
            regret = (
                float(f1_table.loc[class_name, current])
                - float(f1_table.loc[class_name, target])
            )
            candidates.append((regret, class_name, current))

        if not candidates:
            raise RuntimeError(
                "Automatic assignment cannot satisfy minimum expert classes."
            )

        _, class_name, _ = min(candidates)
        owner[class_name] = target

    result = {model: [] for model in MODEL_NAMES}
    for class_name, model in owner.items():
        result[model].append(class_name)

    for model in result:
        result[model].sort(key=class_names.index)
    return result


def auto_hard_negatives(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str],
    expert_classes: list[str],
) -> list[str]:
    labels = list(range(len(class_names)))
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    expert_ids = [
        class_names.index(class_name)
        for class_name in expert_classes
    ]

    candidates = []
    for class_id, class_name in enumerate(class_names):
        if class_name == "background" or class_name in expert_classes:
            continue
        denominator = max(1, int(matrix[class_id].sum()))
        confusion_to_experts = (
            float(matrix[class_id, expert_ids].sum()) / denominator
            if expert_ids
            else 0.0
        )
        candidates.append((confusion_to_experts, class_name))

    candidates.sort(
        key=lambda item: (-item[0], class_names.index(item[1]))
    )
    count = int(config.AUTO_HARD_NEGATIVE_CLASS_COUNT)
    return [name for _, name in candidates[:count]]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ratio", type=float, default=None)
    args = parser.parse_args()

    class_names = read_class_names(config.CLASS_NAMES_PATH)

    validation_files = {
        "efficientat": config.EFFICIENTAT_OUTPUT / "val_predictions.npz",
        "yamnet": config.YAMNET_OUTPUT / "val_predictions.npz",
        "mobilenetv4": config.MOBILENETV4_OUTPUT / "val_predictions.npz",
    }
    _, y_true, _, logits_by_model = align_prediction_files(
        validation_files
    )

    predictions = {}
    f1_values = {}
    for model_name, logits in logits_by_model.items():
        prediction = logits.argmax(axis=1)
        predictions[model_name] = prediction
        f1_values[model_name] = per_class_f1(
            y_true,
            prediction,
            len(class_names),
        )

    f1_table = pd.DataFrame(
        f1_values,
        index=class_names,
    )
    experts = automatic_assignment(f1_table, class_names)
    hard_negatives = {
        model_name: auto_hard_negatives(
            y_true,
            predictions[model_name],
            class_names,
            experts[model_name],
        )
        for model_name in MODEL_NAMES
    }

    metadata = pd.read_csv(config.METADATA_CSV)
    fractions = (
        [float(args.ratio)]
        if args.ratio is not None
        else [float(v) for v in config.SPECIALIST_NON_EXPERT_FRACTIONS]
    )
    fractions = sorted(set(fractions), reverse=True)

    config.AUTO_SPECIALIST_METADATA_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )
    f1_table.to_csv(
        config.AUTO_SPECIALIST_METADATA_ROOT
        / "baseline_validation_per_class_f1.csv"
    )

    all_rows = []
    for fraction in fractions:
        output_dir = config.auto_metadata_dir(fraction)
        output_dir.mkdir(parents=True, exist_ok=True)

        ratio_rows = []
        for model_index, model_name in enumerate(MODEL_NAMES):
            specialist, summary = build_specialist_dataframe(
                metadata=metadata,
                class_names=class_names,
                model_name=model_name,
                expert_classes=experts[model_name],
                hard_negative_classes=hard_negatives[model_name],
                nonexpert_fraction=fraction,
                hard_negative_min_fraction=(
                    config.SPECIALIST_HARD_NEGATIVE_MIN_FRACTION
                ),
                assignment_name="automatic",
                seed=config.SEED + model_index * 10_000,
            )
            specialist.to_csv(
                config.auto_csv(model_name, fraction),
                index=False,
            )
            ratio_rows.extend(summary)
            all_rows.extend(summary)

        assignment = {
            "assignment": "automatic",
            "selection_source": (
                "A/B/C baseline validation per-class F1 only"
            ),
            "ratio_tag": config.ratio_tag(fraction),
            "nonexpert_fraction": fraction,
            "expert_classes": experts,
            "hard_negative_classes": hard_negatives,
            "minimum_expert_classes_per_model": (
                config.AUTO_MIN_EXPERT_CLASSES_PER_MODEL
            ),
            "background_batch_fraction": 0.20,
            "sampling_unit": "group_id",
            "nested_subsets": True,
        }
        (output_dir / "specialist_assignment.json").write_text(
            json.dumps(assignment, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        pd.DataFrame(ratio_rows).to_csv(
            output_dir / "specialist_sampling_summary.csv",
            index=False,
        )
        print(json.dumps(assignment, ensure_ascii=False, indent=2))
        print(f"Wrote automatic: {output_dir}")

    pd.DataFrame(all_rows).to_csv(
        config.AUTO_SPECIALIST_METADATA_ROOT
        / "all_ratio_sampling_summary.csv",
        index=False,
    )


if __name__ == "__main__":
    main()
