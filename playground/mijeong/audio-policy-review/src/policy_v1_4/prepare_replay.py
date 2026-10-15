"""Prepare controlled clip replay from the static pilot's evaluation groups."""
import argparse, json
from pathlib import Path
from collections import Counter
import numpy as np
from project_io import write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="pilot_results")
    parser.add_argument("--selection", action="store_true",
                        help="Use no-voting selection_config.json and selection_samples.npz")
    parser.add_argument("--phase-offsets", type=float, nargs="+", default=None,
                        help="Insertion offsets in seconds; selection default: 0 .5 1 1.5")
    parser.add_argument("--max-per-class", type=int, default=8)
    parser.add_argument("--dataset-root", default=None,
                        help="Optional replacement for the old /.../aug_data root.")
    parser.add_argument("--old-dataset-root", default="/home/plass/YAMNet/aug_data")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.max_per_class < 1:
        raise ValueError("max-per-class must be positive")
    out = Path(args.out).resolve()
    config_file = "selection_config.json" if args.selection else "policy_config.json"
    sample_file = "selection_samples.npz" if args.selection else "static_scores.npz"
    config = json.loads((out / config_file).read_text(encoding="utf-8"))
    phases = args.phase_offsets if args.phase_offsets is not None else ([0., .5, 1., 1.5] if args.selection else [0.])
    if (not phases or len(set(phases)) != len(phases)
            or any(not np.isfinite(x) or not 0 <= x < 2. for x in phases)):
        raise ValueError("phase offsets must be unique, finite and in [0,2)")
    if not args.selection and phases != [0.]:
        raise ValueError("Nonzero phases currently require --selection and evaluate_selection_replay.py")
    with np.load(out / sample_file, allow_pickle=False) as data:
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
            for phase in phases:
                episodes.append({
                    "episode_id": len(episodes), "path": str(path.resolve()),
                    "label": label, "label_id": cid, "group_id": group,
                    "phase_offset_seconds": phase, "clip_insert_seconds": 2. + phase,
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
        "phase_offsets_seconds": phases,
        "config_file": config_file,
        "audio_config": {"highpass_hz": 80.0, "highpass_order": 2,
                         "silence_frame_ms": 100.0, "silence_threshold_dbfs": -30.0,
                         "target_rms_dbfs": -20.0, "max_normalize_gain_db": 20.0},
        "annotation": "Controlled replay: 2s + phase offset silence, original first 2s clip, 2s silence. "
                      "Clip insertion time is known; actual event onset is NOT known. "
                      "Not a natural continuous-audio benchmark. Repeated phases of a clip are NOT independent samples.",
        "selection": "One clip per recorded group from static evaluation groups; each clip repeated at declared phases.",
        "recorded_groups": len(used_groups),
        "class_counts": dict(Counter(e["label"] for e in episodes)),
        "episodes": episodes,
    }
    write_json(out / "replay_manifest.json", manifest)
    print("Prepared", len(episodes), "episodes:", manifest["class_counts"])
    if missing:
        print("Missing source files skipped:", len(missing))


if __name__ == "__main__":
    main()

