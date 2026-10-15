from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

import config
from common_metadata import read_class_names
from evaluation_utils import (
    align_prediction_files,
    per_class_f1,
)


MODEL_NAMES = ["efficientat", "yamnet", "mobilenetv4"]


def _manual_assignment(class_names: list[str]) -> dict[str, list[str]] | None:
    manual = {
        model: list(config.SPECIALIST_MANUAL_EXPERT_CLASSES.get(model, []))
        for model in MODEL_NAMES
    }
    if not any(manual.values()):
        return None

    foreground = set(class_names) - {"background"}
    flat = [label for labels in manual.values() for label in labels]
    unknown = sorted(set(flat) - foreground)
    duplicate = sorted({label for label in flat if flat.count(label) > 1})
    missing = sorted(foreground - set(flat))
    if unknown or duplicate or missing:
        raise ValueError(
            "Invalid SPECIALIST_MANUAL_EXPERT_CLASSES: "
            f"unknown={unknown}, duplicate={duplicate}, missing={missing}"
        )
    return manual


def _automatic_assignment(
    f1_table: pd.DataFrame,
    class_names: list[str],
) -> dict[str, list[str]]:
    foreground = [name for name in class_names if name != "background"]
    assignment = {
        label: str(f1_table.loc[label].idxmax())
        for label in foreground
    }
    minimum = int(config.SPECIALIST_MIN_EXPERT_CLASSES_PER_MODEL)

    # Ensure every architecture receives at least the requested minimum number
    # of expert classes. Move the class with the smallest validation-F1 regret.
    while True:
        counts = {
            model: sum(owner == model for owner in assignment.values())
            for model in MODEL_NAMES
        }
        target = next(
            (model for model in MODEL_NAMES if counts[model] < minimum),
            None,
        )
        if target is None:
            break

        candidates = []
        for label, current in assignment.items():
            if current == target or counts[current] <= minimum:
                continue
            regret = (
                float(f1_table.loc[label, current])
                - float(f1_table.loc[label, target])
            )
            candidates.append((regret, label, current))
        if not candidates:
            raise RuntimeError("Unable to satisfy minimum expert-class assignment")
        _, label, _ = min(candidates)
        assignment[label] = target

    result = {model: [] for model in MODEL_NAMES}
    for label, model in assignment.items():
        result[model].append(label)
    for model in result:
        result[model].sort(key=class_names.index)
    return result


def _hard_negative_classes(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str],
    experts: list[str],
) -> list[str]:
    labels = list(range(len(class_names)))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    expert_ids = [class_names.index(name) for name in experts]
    candidates = []
    for class_id, class_name in enumerate(class_names):
        if class_name == "background" or class_name in experts:
            continue
        denominator = max(1, int(cm[class_id].sum()))
        score = float(cm[class_id, expert_ids].sum()) / denominator
        candidates.append((score, class_name))
    candidates.sort(key=lambda item: (-item[0], class_names.index(item[1])))
    return [
        name
        for _, name in candidates[
            : int(config.SPECIALIST_HARD_NEGATIVE_CLASS_COUNT)
        ]
    ]


def _sample_groups(
    rows: pd.DataFrame,
    fraction: float,
    seed: int,
) -> pd.DataFrame:
    if fraction >= 1.0 or rows.empty:
        return rows.copy()
    groups = sorted(rows["group_id"].astype(str).unique())
    keep_count = max(1, int(math.ceil(len(groups) * fraction)))
    rng = np.random.default_rng(seed)
    keep = set(rng.choice(groups, size=keep_count, replace=False).tolist())
    return rows[rows["group_id"].astype(str).isin(keep)].copy()


def main() -> None:
    class_names = read_class_names(config.CLASS_NAMES_PATH)
    validation_files = {
        "efficientat": config.EFFICIENTAT_OUTPUT / "val_predictions.npz",
        "yamnet": config.YAMNET_OUTPUT / "val_predictions.npz",
        "mobilenetv4": config.MOBILENETV4_OUTPUT / "val_predictions.npz",
    }
    paths, y_true, _, logits_by_model = align_prediction_files(validation_files)

    f1_values = {}
    predictions = {}
    for model, logits in logits_by_model.items():
        pred = logits.argmax(axis=1)
        predictions[model] = pred
        f1_values[model] = per_class_f1(
            y_true,
            pred,
            len(class_names),
        )
    f1_table = pd.DataFrame(
        f1_values,
        index=class_names,
    )

    experts = _manual_assignment(class_names)
    if experts is None:
        experts = _automatic_assignment(f1_table, class_names)

    hard_negatives = {
        model: _hard_negative_classes(
            y_true,
            predictions[model],
            class_names,
            experts[model],
        )
        for model in MODEL_NAMES
    }

    metadata = pd.read_csv(config.METADATA_CSV)
    required = {"path", "label", "split", "group_id"}
    missing = required - set(metadata.columns)
    if missing:
        raise ValueError(f"metadata.csv missing columns: {sorted(missing)}")

    output_dir = Path(config.SPECIALIST_METADATA_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_paths = {
        "efficientat": config.EFFICIENTAT_SPECIALIST_CSV,
        "yamnet": config.YAMNET_SPECIALIST_CSV,
        "mobilenetv4": config.MOBILENETV4_SPECIALIST_CSV,
    }

    summary_rows = []
    for model_index, model in enumerate(MODEL_NAMES):
        train = metadata[metadata["split"].astype(str) == "train"].copy()
        eval_rows = metadata[
            metadata["split"].astype(str).isin(["val", "test"])
        ].copy()

        selected_parts = []
        background = train[train["label"].astype(str) == "background"].copy()
        background["specialist_role"] = "background"
        background["keep_fraction"] = 1.0
        selected_parts.append(background)

        for class_index, class_name in enumerate(class_names):
            if class_name == "background":
                continue
            rows = train[train["label"].astype(str) == class_name].copy()
            if class_name in experts[model]:
                role = "expert"
                fraction = 1.0
            elif class_name in hard_negatives[model]:
                role = "hard_negative"
                fraction = float(config.SPECIALIST_HARD_NEGATIVE_FRACTION)
            else:
                role = "ordinary_non_expert"
                fraction = float(config.SPECIALIST_NON_EXPERT_FRACTION)

            kept = _sample_groups(
                rows,
                fraction,
                seed=int(config.SEED + model_index * 10_000 + class_index),
            )
            kept["specialist_role"] = role
            kept["keep_fraction"] = fraction
            selected_parts.append(kept)
            summary_rows.append(
                {
                    "model": model,
                    "label": class_name,
                    "role": role,
                    "keep_fraction": fraction,
                    "original_train_rows": len(rows),
                    "selected_train_rows": len(kept),
                    "original_groups": rows["group_id"].astype(str).nunique(),
                    "selected_groups": kept["group_id"].astype(str).nunique(),
                    "baseline_val_f1": float(
                        f1_table.loc[class_name, model]
                    ),
                }
            )

        eval_rows["specialist_role"] = "evaluation_full_distribution"
        eval_rows["keep_fraction"] = 1.0
        specialist = pd.concat(
            selected_parts + [eval_rows],
            ignore_index=True,
        )
        specialist["specialist_model"] = model
        specialist.to_csv(output_paths[model], index=False)

    f1_table.to_csv(output_dir / "baseline_validation_per_class_f1.csv")
    pd.DataFrame(summary_rows).to_csv(
        output_dir / "specialist_sampling_summary.csv",
        index=False,
    )
    assignment = {
        "expert_classes": experts,
        "hard_negative_classes": hard_negatives,
        "policy": {
            "expert_fraction": 1.0,
            "hard_negative_fraction": config.SPECIALIST_HARD_NEGATIVE_FRACTION,
            "ordinary_non_expert_fraction": config.SPECIALIST_NON_EXPERT_FRACTION,
            "background_batch_fraction": 0.20,
            "selection_basis": "baseline validation per-class F1 and confusion",
        },
    }
    (output_dir / "specialist_assignment.json").write_text(
        json.dumps(assignment, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(assignment, ensure_ascii=False, indent=2))
    for model, path in output_paths.items():
        print(f"{model}: {path}")


if __name__ == "__main__":
    main()
