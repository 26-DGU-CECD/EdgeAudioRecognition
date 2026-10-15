from __future__ import annotations

import hashlib
import math

import pandas as pd


def stable_group_rank(
    group_id: str,
    assignment_name: str,
    model_name: str,
    class_name: str,
    seed: int,
) -> str:
    key = (
        f"{seed}|{assignment_name}|{model_name}|{class_name}|{group_id}"
    ).encode("utf-8")
    return hashlib.sha256(key).hexdigest()


def nested_group_subset(
    rows: pd.DataFrame,
    fraction: float,
    assignment_name: str,
    model_name: str,
    class_name: str,
    seed: int,
) -> pd.DataFrame:
    """Low-ratio sets are deterministic subsets of high-ratio sets."""
    if rows.empty or fraction >= 1.0:
        return rows.copy()

    groups = sorted(rows["group_id"].astype(str).unique())
    ranked = sorted(
        groups,
        key=lambda group: stable_group_rank(
            group,
            assignment_name,
            model_name,
            class_name,
            seed,
        ),
    )
    keep_count = max(1, int(math.ceil(len(ranked) * fraction)))
    keep_groups = set(ranked[:keep_count])
    return rows[
        rows["group_id"].astype(str).isin(keep_groups)
    ].copy()


def build_specialist_dataframe(
    metadata: pd.DataFrame,
    class_names: list[str],
    model_name: str,
    expert_classes: list[str],
    hard_negative_classes: list[str],
    nonexpert_fraction: float,
    hard_negative_min_fraction: float,
    assignment_name: str,
    seed: int,
) -> tuple[pd.DataFrame, list[dict]]:
    train = metadata[
        metadata["split"].astype(str) == "train"
    ].copy()
    evaluation = metadata[
        metadata["split"].astype(str).isin(["val", "test"])
    ].copy()

    selected_parts: list[pd.DataFrame] = []
    summary_rows: list[dict] = []

    background = train[
        train["label"].astype(str) == "background"
    ].copy()
    background["specialist_role"] = "background"
    background["keep_fraction"] = 1.0
    selected_parts.append(background)

    for class_index, class_name in enumerate(class_names):
        if class_name == "background":
            continue

        rows = train[
            train["label"].astype(str) == class_name
        ].copy()

        if class_name in expert_classes:
            role = "expert"
            keep_fraction = 1.0
        elif class_name in hard_negative_classes:
            role = "hard_negative"
            keep_fraction = max(
                float(nonexpert_fraction),
                float(hard_negative_min_fraction),
            )
        else:
            role = "ordinary_non_expert"
            keep_fraction = float(nonexpert_fraction)

        kept = nested_group_subset(
            rows=rows,
            fraction=keep_fraction,
            assignment_name=assignment_name,
            model_name=model_name,
            class_name=class_name,
            seed=seed + class_index,
        )
        kept["specialist_role"] = role
        kept["keep_fraction"] = keep_fraction
        selected_parts.append(kept)

        summary_rows.append(
            {
                "assignment": assignment_name,
                "requested_nonexpert_fraction": nonexpert_fraction,
                "model": model_name,
                "label": class_name,
                "role": role,
                "applied_keep_fraction": keep_fraction,
                "original_train_rows": len(rows),
                "selected_train_rows": len(kept),
                "original_groups": rows["group_id"].astype(str).nunique(),
                "selected_groups": kept["group_id"].astype(str).nunique(),
            }
        )

    evaluation["specialist_role"] = "evaluation_full_distribution"
    evaluation["keep_fraction"] = 1.0

    specialist = pd.concat(
        selected_parts + [evaluation],
        ignore_index=True,
    )
    specialist["specialist_model"] = model_name
    specialist["assignment"] = assignment_name
    specialist["requested_nonexpert_fraction"] = nonexpert_fraction
    return specialist, summary_rows
