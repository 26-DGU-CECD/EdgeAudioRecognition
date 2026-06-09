from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from visualize import save_html_report


SCRIPT_DIR = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a MobileNetV4-small ESC-50 HTML report from saved CSV/JSON/PNG results."
    )
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    return parser.parse_args()


def resolve_results_dir(path: Path) -> Path:
    if path.is_absolute():
        return path.expanduser().resolve()
    return (SCRIPT_DIR / path).resolve()


def require_file(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Required report input not found: {path}")
    return path


def build_report(results_dir: Path) -> Path:
    results_dir = resolve_results_dir(results_dir)
    plots_dir = results_dir / "plots"

    overall_metrics = json.loads(
        require_file(results_dir / "overall_metrics.json").read_text(encoding="utf-8")
    )
    overall_df = pd.read_csv(require_file(results_dir / "overall_metrics.csv"))
    per_category_df = pd.read_csv(require_file(results_dir / "per_category_metrics.csv"))
    top_confusions_df = pd.read_csv(require_file(results_dir / "top_confusions.csv"))

    return save_html_report(
        overall_metrics=overall_metrics,
        overall_df=overall_df,
        per_category_df=per_category_df,
        top_confusions_df=top_confusions_df,
        plots_dir=plots_dir,
        output_dir=results_dir,
    )


def main() -> int:
    args = parse_args()
    try:
        report_path = build_report(args.results_dir)
    except (FileNotFoundError, ValueError, json.JSONDecodeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(f"Saved HTML report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
