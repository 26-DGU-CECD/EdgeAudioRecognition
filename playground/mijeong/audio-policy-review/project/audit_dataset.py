from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from common_audio import AudioConfig, load_and_preprocess
import config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default=str(config.METADATA_CSV))
    parser.add_argument("--output", default=str(config.DATASET_AUDIT_CSV))
    parser.add_argument("--highpass-hz", type=float, default=config.HIGHPASS_HZ)
    parser.add_argument("--silence-threshold-dbfs", type=float, default=config.SILENCE_THRESHOLD_DBFS)
    parser.add_argument("--target-rms-dbfs", type=float, default=config.TARGET_RMS_DBFS)
    args = parser.parse_args()
    config.validate_paths()

    cfg = AudioConfig(
        highpass_hz=args.highpass_hz,
        silence_threshold_dbfs=args.silence_threshold_dbfs,
        target_rms_dbfs=args.target_rms_dbfs,
    )
    csv_path = Path(args.csv).resolve()
    df = pd.read_csv(csv_path)
    base = csv_path.parent
    rows = []
    for i, row in df.iterrows():
        p = Path(str(row["path"]))
        if not p.is_absolute():
            p = (base / p).resolve()
        _, is_silent, max_dbfs = load_and_preprocess(p, cfg)
        rows.append(
            {
                **row.to_dict(),
                "resolved_path": str(p),
                "pipeline_is_silent": bool(is_silent),
                "max_100ms_rms_dbfs_after_highpass": float(max_dbfs),
                "label_mismatch": bool((row["label"] == "background") != bool(is_silent)),
            }
        )
        if (i + 1) % 500 == 0:
            print(f"audited {i + 1}/{len(df)}")
    out = pd.DataFrame(rows)
    out.to_csv(args.output, index=False)
    print(out.groupby(["label", "pipeline_is_silent"]).size().unstack(fill_value=0).to_string())
    print(f"Wrote: {Path(args.output).resolve()}")
    print(
        "Note: background/non-background mismatch is an audit flag, not an automatic relabel command. "
        "Background may later include non-silent hard negatives."
    )


if __name__ == "__main__":
    main()
