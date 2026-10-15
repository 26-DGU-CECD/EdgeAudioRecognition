from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

import config


def read_metrics(path: Path) -> dict | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def append_system(
    rows: list[dict],
    assignment: str,
    fraction: float,
    system: str,
    metrics_path: Path,
) -> None:
    metrics = read_metrics(metrics_path)
    if metrics is None:
        return
    rows.append(
        {
            "assignment": assignment,
            "nonexpert_fraction": fraction,
            "ratio_percent": fraction * 100.0,
            "system": system,
            "accuracy": metrics["accuracy"],
            "macro_f1": metrics["macro_f1"],
        }
    )


def main() -> None:
    rows: list[dict] = []
    fractions = sorted(
        {float(v) for v in config.SPECIALIST_NON_EXPERT_FRACTIONS},
        reverse=True,
    )

    for fraction in fractions:
        for model_name in config.MODEL_NAMES:
            append_system(
                rows,
                "predefined",
                fraction,
                f"E_predefined_{model_name}",
                config.predefined_output(model_name, fraction)
                / "test_metrics.json",
            )
            append_system(
                rows,
                "automatic",
                fraction,
                f"G_auto_{model_name}",
                config.auto_output(model_name, fraction)
                / "test_metrics.json",
            )

        append_system(
            rows,
            "predefined",
            fraction,
            "F_predefined_weighted_ensemble",
            config.predefined_ensemble_output(fraction)
            / "test_metrics.json",
        )
        append_system(
            rows,
            "automatic",
            fraction,
            "G_auto_weighted_ensemble",
            config.auto_ensemble_output(fraction)
            / "test_metrics.json",
        )

    if not rows:
        raise FileNotFoundError(
            "No E/F/G specialist test metrics were found."
        )

    output = Path(config.SPECIALIST_SWEEP_SUMMARY)
    output.mkdir(parents=True, exist_ok=True)

    frame = pd.DataFrame(rows).sort_values(
        ["assignment", "system", "nonexpert_fraction"],
        ascending=[True, True, False],
    )
    frame.to_csv(
        output / "specialist_ratio_results.csv",
        index=False,
    )

    for metric in ["macro_f1", "accuracy"]:
        fig, ax = plt.subplots(figsize=(11, 7))
        for system, group in frame.groupby("system"):
            ordered = group.sort_values("ratio_percent")
            ax.plot(
                ordered["ratio_percent"],
                ordered[metric],
                marker="o",
                label=system,
            )
        ax.set_xlabel("Ordinary non-expert training data (%)")
        ax.set_ylabel(metric)
        ax.set_title(f"E/F/G specialist ratio sweep: {metric}")
        ax.set_xticks(sorted(frame["ratio_percent"].unique()))
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(
            output / f"{metric}_by_nonexpert_ratio.png",
            dpi=200,
            bbox_inches="tight",
        )
        plt.close(fig)

    # Direct predefined-vs-auto ensemble comparison.
    ensemble_frame = frame[
        frame["system"].isin(
            [
                "F_predefined_weighted_ensemble",
                "G_auto_weighted_ensemble",
            ]
        )
    ].copy()
    ensemble_frame.to_csv(
        output / "predefined_vs_auto_ensemble.csv",
        index=False,
    )

    print(frame.to_string(index=False))
    print(f"\nWrote: {output}")


if __name__ == "__main__":
    main()
