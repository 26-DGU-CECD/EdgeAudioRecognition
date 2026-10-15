from __future__ import annotations

import argparse
import json

import pandas as pd

import config
from common_metadata import read_class_names
from specialist_data_utils import build_specialist_dataframe


MODEL_NAMES = list(config.MODEL_NAMES)


def validate_predefined_classes(
    class_names: list[str],
) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    foreground = set(class_names) - {"background"}
    experts = {
        model: list(config.SPECIALIST_EXPERT_CLASSES.get(model, []))
        for model in MODEL_NAMES
    }
    hard_negatives = {
        model: list(config.SPECIALIST_HARD_NEGATIVE_CLASSES.get(model, []))
        for model in MODEL_NAMES
    }

    flat = [
        class_name
        for model in MODEL_NAMES
        for class_name in experts[model]
    ]
    unknown = sorted(set(flat) - foreground)
    duplicate = sorted(
        {
            class_name
            for class_name in flat
            if flat.count(class_name) > 1
        }
    )
    missing = sorted(foreground - set(flat))

    if unknown or duplicate or missing:
        raise ValueError(
            "SPECIALIST_EXPERT_CLASSES 설정 오류\n"
            f"unknown={unknown}\n"
            f"duplicate={duplicate}\n"
            f"missing={missing}\n"
            "background를 제외한 13개 클래스를 정확히 한 모델에만 배정하세요."
        )

    for model in MODEL_NAMES:
        unknown_hard = sorted(set(hard_negatives[model]) - foreground)
        overlap = sorted(set(hard_negatives[model]) & set(experts[model]))
        if unknown_hard or overlap:
            raise ValueError(
                f"{model} hard-negative 오류: "
                f"unknown={unknown_hard}, expert_overlap={overlap}"
            )
    return experts, hard_negatives


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ratio", type=float, default=None)
    args = parser.parse_args()

    class_names = read_class_names(config.CLASS_NAMES_PATH)
    experts, hard_negatives = validate_predefined_classes(class_names)
    metadata = pd.read_csv(config.METADATA_CSV)

    fractions = (
        [float(args.ratio)]
        if args.ratio is not None
        else [float(v) for v in config.SPECIALIST_NON_EXPERT_FRACTIONS]
    )
    fractions = sorted(set(fractions), reverse=True)

    all_rows = []
    for fraction in fractions:
        output_dir = config.predefined_metadata_dir(fraction)
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
                assignment_name="predefined",
                seed=config.SEED + model_index * 10_000,
            )
            specialist.to_csv(
                config.predefined_csv(model_name, fraction),
                index=False,
            )
            ratio_rows.extend(summary)
            all_rows.extend(summary)

        assignment = {
            "assignment": "predefined",
            "ratio_tag": config.ratio_tag(fraction),
            "nonexpert_fraction": fraction,
            "expert_classes": experts,
            "hard_negative_classes": hard_negatives,
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
        print(f"Wrote predefined: {output_dir}")

    config.PREDEFINED_SPECIALIST_METADATA_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )
    pd.DataFrame(all_rows).to_csv(
        config.PREDEFINED_SPECIALIST_METADATA_ROOT
        / "all_ratio_sampling_summary.csv",
        index=False,
    )


if __name__ == "__main__":
    main()
