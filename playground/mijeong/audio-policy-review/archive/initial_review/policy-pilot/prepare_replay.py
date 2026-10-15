"""Prepare controlled clip replay from the static pilot's evaluation groups."""
import argparse, json
from pathlib import Path
from collections import Counter
import numpy as np
from project_io import write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="pilot_results")
    parser.add_argument("--max-per-class", type=int, default=8)
    parser.add_argument("--dataset-root", default=None,
                        help="Optional replacement for the old /.../aug_data root.")
    parser.add_argument("--old-dataset-root", default="/home/plass/YAMNet/aug_data")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.max_per_class < 1:
        raise ValueError("max-per-class must be positive")
    out = Path(args.out).resolve()
    config = json.loads((out / "policy_config.json").read_text(encoding="utf-8"))
    with np.load(out / "static_scores.npz", allow_pickle=False) as data:
        mask = data["evaluation_mask"]
        paths = data["paths"][mask].astype(str)
        labels = data["y_true"][mask]
        groups = data["group_ids"][mask].astype(str)
    rng = np.random.default_rng(args.seed)
    episodes = []
    used_groups = set()
    missing = []
    for cid, label in enumerate(config["class_names"]):
        candidates = np.flatnonzero(labels == cid)
        candidates = candidates[rng.permutation(len(candidates))]
        kept = 0
        for idx in candidates:
            group = groups[idx]
            if group in used_groups:
                continue
            path = Path(paths[idx])
            if args.dataset_root:
                try:
                    relative = path.relative_to(Path(args.old_dataset_root))
                except ValueError:
                    raise ValueError(
                        f"{path} is not under --old-dataset-root; specify its original root")
                path = Path(args.dataset_root) / relative
            if not path.is_file():
                missing.append(str(path))
                continue
            used_groups.add(group)
            episodes.append({
                "episode_id": len(episodes), "path": str(path.resolve()),
                "label": label, "label_id": cid, "group_id": group,
            })
            kept += 1
            if kept >= args.max_per_class:
                break
    expected_labels = set(config["class_names"])
    represented = {e["label"] for e in episodes}
    if represented != expected_labels:
        raise FileNotFoundError(
            f"Replay missing classes {sorted(expected_labels-represented)}; "
            f"missing file examples={missing[:3]}. Check dataset-root mapping.")
    manifest = {
        "sample_rate": 16000, "window_seconds": 2.0,
        "base_hop_seconds": 0.5, "warmup_seconds": 2.0, "tail_seconds": 2.0,
        "input_policies": {"I0": 2.0, "I1": 1.0, "I2": 0.5},
        "audio_config": {"highpass_hz": 80.0, "highpass_order": 2,
                         "silence_frame_ms": 100.0, "silence_threshold_dbfs": -30.0,
                         "target_rms_dbfs": -20.0, "max_normalize_gain_db": 20.0},
        "annotation": "Controlled replay: 2s silence + original first 2s clip + 2s silence. "
                      "Clip insertion time is known; actual event onset is NOT known. "
                      "Not a natural continuous-audio benchmark.",
        "selection": "One clip per recorded group, only static evaluation groups.",
        "class_counts": dict(Counter(e["label"] for e in episodes)),
        "episodes": episodes,
    }
    write_json(out / "replay_manifest.json", manifest)
    print("Prepared", len(episodes), "episodes:", manifest["class_counts"])
    if missing:
        print("Missing source files skipped:", len(missing))


if __name__ == "__main__":
    main()

