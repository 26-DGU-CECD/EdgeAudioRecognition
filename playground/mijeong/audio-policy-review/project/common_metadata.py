from __future__ import annotations

from pathlib import Path
from typing import Sequence

import pandas as pd


def read_class_names(path: str | Path) -> list[str]:
    names = [
        line.strip()
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(names) != len(set(names)):
        raise ValueError("class_names.txt contains duplicate labels")
    if "background" not in names:
        raise ValueError("class_names.txt must include 'background'")
    return names


def load_metadata(csv_path: str | Path, split: str) -> pd.DataFrame:
    csv_path = Path(csv_path).resolve()
    df = pd.read_csv(csv_path)
    required = {"path", "label", "split", "group_id"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"metadata CSV is missing columns: {sorted(missing)}")

    df = df[df["split"].astype(str) == split].copy()
    base = csv_path.parent
    df["path"] = df["path"].map(
        lambda p: str((base / str(p)).resolve())
        if not Path(str(p)).is_absolute()
        else str(Path(str(p)).resolve())
    )
    if df.empty:
        raise ValueError(f"No rows for split={split!r}")
    return df.reset_index(drop=True)
