"""Compare 3 input hops x 3 output rules on cached controlled-replay logits."""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
import numpy as np
from policy_core import INPUT_POLICIES, MODEL_NAMES, OutputPolicy, fuse_logits
from project_io import write_csv, write_json


def policies_from_config(config):
    names = config["class_names"]
    return [
        OutputPolicy(names, "O0"),
        OutputPolicy(names, "O1", config["global_threshold"]),
        OutputPolicy(names, "O2", class_thresholds=config["class_thresholds"]),
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="pilot_results")
    args = parser.parse_args()
    out = Path(args.out).resolve()
    cfg_bytes = (out / "policy_config.json").read_bytes()
    manifest_bytes = (out / "replay_manifest.json").read_bytes()
    config, manifest = json.loads(cfg_bytes), json.loads(manifest_bytes)
    fingerprint = hashlib.sha256(cfg_bytes + manifest_bytes).hexdigest()
    canonical, arrays, service = None, [], []
    for model in MODEL_NAMES:
        with np.load(out / f"replay_{model}.npz", allow_pickle=False) as data:
            if str(data["fingerprint"].item()) != fingerprint:
                raise ValueError(f"{model}: stale export; config/manifest changed")
            keys = data["keys"].astype(str)
            if len(set(keys.tolist())) != len(keys):
                raise ValueError("Duplicate window keys")
            if canonical is None:
                canonical = keys
                order = np.arange(len(keys))
            else:
                index = {k: i for i, k in enumerate(keys)}
                if set(index) != set(canonical.tolist()):
                    raise ValueError("Model export windows differ")
                order = np.array([index[k] for k in canonical])
            arrays.append(data["logits"][order])
            service.append((data["preprocessing_s"] + data["inference_s"])[order])
    p = fuse_logits(np.stack(arrays), np.asarray(config["weights"]))
    service = np.stack(service)
    sr = int(manifest["sample_rate"])
    window_n = round(manifest["window_seconds"] * sr)
    base_hop_n = round(manifest["base_hop_seconds"] * sr)
    insert_start = round(manifest["warmup_seconds"] * sr)
    insert_end = insert_start + window_n
    total_samples = insert_end + round(manifest["tail_seconds"] * sr)
    expected = {
        f"{episode['episode_id']}:{end}"
        for episode in manifest["episodes"]
        for end in range(window_n, total_samples + 1, base_hop_n)
    }
    if set(canonical.tolist()) != expected:
        raise ValueError("Incomplete replay exports")
    episode_ids = np.array([int(k.split(":")[0]) for k in canonical])
    ends = np.array([int(k.split(":")[1]) for k in canonical])
    starts = ends - window_n
    details, summary = [], []
    for input_policy, hop_s in INPUT_POLICIES.items():
        hop_n = round(hop_s * sr)
        selected = (ends - window_n) % hop_n == 0
        for policy in policies_from_config(config):
            decision = policy.decide(p)
            foreground_hits, first_hits, bg_flags, wrong_flags = [], [], [], []
            for episode in manifest["episodes"]:
                # A window counts as having proxy clip evidence only if it
                # intersects the inserted clip; actual event onset is unknown.
                relevant = selected & (episode_ids == episode["episode_id"]) \
                           & (ends > insert_start) & (starts < insert_end)
                idx = np.flatnonzero(relevant)
                idx = idx[np.argsort(ends[idx])]
                accepted = decision["accepted"][idx]
                labels = decision["top_id"][idx]
                is_background = episode["label"] == "background"
                match = accepted & (labels == episode["label_id"]) & (not is_background)
                times = (ends[idx] - insert_start) / sr
                first = float(times[match][0]) if match.any() else None
                by_two = bool(np.any(match & (times <= 2.0)))
                wrong = bool(np.any(accepted & (labels != episode["label_id"])))
                details.append({
                    "input_policy": input_policy, "output_policy": policy.kind,
                    "episode_id": episode["episode_id"], "label": episode["label"],
                    "group_id": episode["group_id"],
                    "first_correct_audio_time_from_insert_s": first,
                    "correct_by_insert_2s_audio_time": by_two,
                    "any_wrong_label": wrong,
                    "background_any_alert": bool(accepted.any()) if is_background else None,
                })
                if is_background:
                    bg_flags.append(bool(accepted.any()))
                else:
                    foreground_hits.append(by_two)
                    wrong_flags.append(wrong)
                    if first is not None:
                        first_hits.append(first)
            summary.append({
                "input_policy": input_policy, "hop_seconds": hop_s,
                "output_policy": policy.kind,
                "foreground_episodes": len(foreground_hits),
                "foreground_hit_by_insert_2s_audio_time": float(np.mean(foreground_hits))
                    if foreground_hits else None,
                "foreground_missed_by_insert_2s": sum(not x for x in foreground_hits),
                "first_hit_audio_time_s_p50_detected_only": float(np.median(first_hits))
                    if first_hits else None,
                "first_hit_audio_time_s_p95_detected_only": float(np.quantile(first_hits, .95))
                    if first_hits else None,
                "background_episode_any_alert_rate": float(np.mean(bg_flags))
                    if bg_flags else None,
                "foreground_episode_any_wrong_label_rate": float(np.mean(wrong_flags))
                    if wrong_flags else None,
                "scheduled_windows": int(selected.sum()),
                "scheduled_model_calls": int(selected.sum() * len(MODEL_NAMES)),
                "cached_serial_service_s": float(service[:, selected].sum()),
                "cached_step_serial_service_ms_p95": float(np.quantile(
                    service[:, selected].sum(axis=0), .95) * 1000),
            })
    experiment_id = config.get("experiment_spec", {}).get(
        "experiment_id", config.get("assignment", "legacy"))
    for row in summary + details:
        row.update(experiment_id=experiment_id,
                   policy_parameter_mode=config.get("policy_parameter_mode", "legacy"))
    write_csv(out / "replay_policy_comparison.csv", summary)
    write_csv(out / "replay_per_episode.csv", details)
    write_json(out / "replay_scope.json", {
        "annotation": manifest["annotation"],
        "limitations": [
            "Audio-time hit is relative to clip insertion, not true event onset or real UI latency.",
            "Only one original clip per episode; not overlapping real-world events.",
            "The same base-hop cache is subsetted for all policies.",
            "Service cost uses isolated-worker profiling; no simultaneous GPU queue/IPC/UI measurement.",
            "O2 thresholds came from static clips and are not yet calibrated for rolling windows.",
        ],
    })
    for row in summary:
        print(row)
    print("Saved comparison:", out / "replay_policy_comparison.csv")


if __name__ == "__main__":
    main()

