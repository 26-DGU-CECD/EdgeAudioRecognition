from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)


def stable_softmax(x: np.ndarray, axis: int = 1) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    shifted = x - np.max(x, axis=axis, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.clip(exp.sum(axis=axis, keepdims=True), 1e-12, None)


def load_prediction_file(path: str | Path) -> dict[str, np.ndarray]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Prediction file not found: {path}\n"
            "Export validation/test predictions before running the ensemble."
        )
    data = np.load(path, allow_pickle=True)
    required = {"y_true", "logits", "paths"}
    missing = required - set(data.files)
    if missing:
        raise ValueError(f"{path} is missing arrays: {sorted(missing)}")
    result = {key: data[key] for key in data.files}
    result["paths"] = np.asarray(result["paths"]).astype(str)
    result["y_true"] = np.asarray(result["y_true"], dtype=np.int64)
    result["logits"] = np.asarray(result["logits"], dtype=np.float64)
    return result


def align_prediction_files(
    files: Mapping[str, str | Path],
) -> tuple[list[str], np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    loaded = {name: load_prediction_file(path) for name, path in files.items()}
    first_name = next(iter(loaded))
    canonical_paths = loaded[first_name]["paths"].tolist()
    if len(canonical_paths) != len(set(canonical_paths)):
        raise ValueError(f"Duplicate paths found in {files[first_name]}")
    canonical_true = loaded[first_name]["y_true"]
    aligned_logits: dict[str, np.ndarray] = {}

    for name, item in loaded.items():
        index = {path: i for i, path in enumerate(item["paths"].tolist())}
        missing = [path for path in canonical_paths if path not in index]
        extras = [path for path in index if path not in set(canonical_paths)]
        if missing or extras:
            raise ValueError(
                f"Prediction path mismatch for {name}: "
                f"missing={len(missing)}, extras={len(extras)}"
            )
        order = np.asarray([index[path] for path in canonical_paths], dtype=np.int64)
        y_true = item["y_true"][order]
        if not np.array_equal(y_true, canonical_true):
            raise ValueError(f"y_true mismatch after path alignment for {name}")
        aligned_logits[name] = item["logits"][order]

    return canonical_paths, canonical_true, np.arange(len(canonical_true)), aligned_logits


def per_class_f1(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_classes: int,
) -> np.ndarray:
    return f1_score(
        y_true,
        y_pred,
        labels=list(range(num_classes)),
        average=None,
        zero_division=0,
    )


def plot_confusion_matrices(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Sequence[str],
    output_dir: str | Path,
    prefix: str,
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    labels = list(range(len(class_names)))
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    def draw(matrix: np.ndarray, filename: str, title: str, value_format: str) -> None:
        fig, ax = plt.subplots(figsize=(13, 11))
        image = ax.imshow(matrix, interpolation="nearest", aspect="auto")
        fig.colorbar(image, ax=ax)
        ax.set_title(title)
        ax.set_xlabel("Predicted label")
        ax.set_ylabel("True label")
        ticks = np.arange(len(class_names))
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        ax.set_xticklabels(class_names, rotation=45, ha="right")
        ax.set_yticklabels(class_names)

        threshold = float(matrix.max()) / 2.0 if matrix.size else 0.0
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                value = matrix[i, j]
                text = format(value, value_format)
                ax.text(
                    j,
                    i,
                    text,
                    ha="center",
                    va="center",
                    fontsize=7,
                    color="white" if value > threshold else "black",
                )
        fig.tight_layout()
        fig.savefig(output_dir / filename, dpi=200, bbox_inches="tight")
        plt.close(fig)

    draw(
        cm.astype(np.int64),
        f"{prefix}_confusion_matrix.png",
        f"{prefix.capitalize()} Confusion Matrix",
        "d",
    )
    row_sum = cm.sum(axis=1, keepdims=True)
    normalized = np.divide(
        cm.astype(np.float64),
        row_sum,
        out=np.zeros_like(cm, dtype=np.float64),
        where=row_sum != 0,
    )
    draw(
        normalized,
        f"{prefix}_confusion_matrix_normalized.png",
        f"{prefix.capitalize()} Confusion Matrix (Row-normalized)",
        ".2f",
    )


def save_classification_artifacts(
    output_dir: str | Path,
    split: str,
    y_true: np.ndarray,
    scores: np.ndarray,
    paths: Sequence[str],
    class_names: Sequence[str],
    extra_arrays: Mapping[str, np.ndarray] | None = None,
) -> dict:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    y_true = np.asarray(y_true, dtype=np.int64)
    scores = np.asarray(scores)
    y_pred = scores.argmax(axis=1)

    report = classification_report(
        y_true,
        y_pred,
        labels=list(range(len(class_names))),
        target_names=list(class_names),
        zero_division=0,
        output_dict=True,
    )
    (output_dir / f"{split}_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    arrays = {
        "y_true": y_true,
        "y_pred": y_pred,
        "logits": scores,
        "paths": np.asarray(paths),
    }
    if extra_arrays:
        arrays.update(extra_arrays)
    np.savez_compressed(output_dir / f"{split}_predictions.npz", **arrays)

    metrics = {
        "split": split,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(
            f1_score(y_true, y_pred, average="macro", zero_division=0)
        ),
    }
    (output_dir / f"{split}_metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    plot_confusion_matrices(
        y_true,
        y_pred,
        class_names,
        output_dir,
        prefix=split,
    )
    print(
        f"{split} accuracy={metrics['accuracy']:.4f}, "
        f"macro_f1={metrics['macro_f1']:.4f}"
    )
    return report
