"""Input-hop experiment using owner-only max scores, with no voting.

Uses isolated-worker replay caches. Audio-time proxy is clip insertion time,
not annotated event onset, real-time completion, or measured display latency.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from policy_core import INPUT_POLICIES
from selection_core import BACKGROUND_MODES, owned_candidates, decide
from project_io import write_csv, write_json


def load_exports(out, config, fingerprint):
    spec = config["selection_spec"]
    names = spec["model_class_names"]
    canonical, outputs, services, provenance = None, {}, {}, {}
    key = "logits" if spec["score_type"] == "softmax_logits" else "probabilities"
    for model in spec["expert_classes"]:
        with np.load(out / f"replay_{model}.npz", allow_pickle=False) as data:
            if str(data["fingerprint"].item()) != fingerprint:
                raise ValueError(f"{model}: stale export; config or manifest changed")
            if data["head_class_names"].astype(str).tolist() != names[model]:
                raise ValueError(f"{model}: head label order differs")
            keys = data["keys"].astype(str)
            if len(set(keys.tolist())) != len(keys):
                raise ValueError("duplicate export keys")
            if canonical is None:
                canonical = keys
                order = np.arange(len(keys))
            else:
                lookup = {k: i for i, k in enumerate(keys)}
                if set(lookup) != set(canonical.tolist()):
                    raise ValueError("different windows across specialists")
                order = np.array([lookup[k] for k in canonical])
            outputs[model] = data[key][order]
            service = (data["preprocessing_s"] + data["inference_s"])[order]
            if service.shape != (len(keys),) or not np.isfinite(service).all() or (service < 0).any():
                raise ValueError("invalid isolated-worker service timings")
            services[model] = service
            provenance[model] = str(data["checkpoint_sha256"].item())
    return canonical, outputs, services, provenance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="selection_reference")
    parser.add_argument("--background-mode", choices=(*BACKGROUND_MODES, "all"), default="all")
    parser.add_argument("--output-policies", nargs="+", choices=("S0", "S1", "S2"), default=["S0"],
                        help="Keep S0 fixed to isolate input policy; add S1 S2 for factorial comparison")
    args = parser.parse_args()
    out = Path(args.out).resolve()
    config_bytes = (out / "selection_config.json").read_bytes()
    manifest_bytes = (out / "replay_manifest.json").read_bytes()
    config, manifest = json.loads(config_bytes), json.loads(manifest_bytes)
    if manifest.get("config_file") != "selection_config.json":
        raise ValueError("prepare_replay.py --selection is required")
    fingerprint = hashlib.sha256(config_bytes + manifest_bytes).hexdigest()
    keys, outputs, services, checkpoint_hashes = load_exports(out, config, fingerprint)
    sr = int(manifest["sample_rate"])
    window_n = round(manifest["window_seconds"] * sr)
    base_hop = round(manifest["base_hop_seconds"] * sr)
    if window_n < 1 or base_hop < 1:
        raise ValueError("invalid replay grid")
    episodes = manifest["episodes"]
    if len({e["episode_id"] for e in episodes}) != len(episodes):
        raise ValueError("duplicate episode IDs")
    expected, duration = set(), 0.
    for e in episodes:
        insert = float(e.get("clip_insert_seconds", manifest["warmup_seconds"]))
        total = round((insert + manifest["window_seconds"] + manifest["tail_seconds"]) * sr)
        expected.update(f"{e['episode_id']}:{end}" for end in range(window_n, total + 1, base_hop))
        duration += total / sr
        if e["label"] != config["class_names"][e["label_id"]]:
            raise ValueError("episode label IDs differ from global class names")
    if set(keys.tolist()) != expected:
        raise ValueError("missing, extra or future replay windows")
    episode_ids = np.array([int(k.split(":")[0]) for k in keys])
    ends = np.array([int(k.split(":")[1]) for k in keys])
    starts = ends - window_n
    modes = BACKGROUND_MODES if args.background_mode == "all" else (args.background_mode,)
    details, summary, worker_rows = [], [], []
    for mode in modes:
        if mode not in config["policies"]:
            raise ValueError(f"No fitted thresholds for background mode {mode}; rerun static selection")
        fitted = config["policies"][mode]
        candidates = owned_candidates(outputs, config["class_names"],
                                      config["selection_spec"]["expert_classes"],
                                      config["selection_spec"]["model_class_names"],
                                      mode, config["selection_spec"]["score_type"])
        for kind in args.output_policies:
            decision = decide(candidates, kind, fitted["global_threshold"], fitted["class_thresholds"])
            for input_name, hop in INPUT_POLICIES.items():
                hop_n = round(hop * sr)
                if hop_n % base_hop:
                    raise ValueError("input hop is not represented on cache grid")
                selected = (ends - window_n) % hop_n == 0
                local = []
                for e in episodes:
                    insert_n = round(e.get("clip_insert_seconds", manifest["warmup_seconds"]) * sr)
                    insert_end = insert_n + window_n
                    same_episode = selected & (episode_ids == e["episode_id"])
                    evidence = same_episode & (ends > insert_n) & (starts < insert_end)
                    idx = np.flatnonzero(evidence)
                    idx = idx[np.argsort(ends[idx])]
                    alert = decision["accepted"][idx]
                    labels = decision["top_id"][idx]
                    is_background = e["label"] == "background"
                    match = alert & (labels == e["label_id"]) & (not is_background)
                    times = (ends[idx] - insert_n) / sr
                    first = float(times[match][0]) if match.any() else None
                    # Also count alerts in synthetic pre-clip silence; do not
                    # silently discard that interval from the diagnostic.
                    prefix = same_episode & (ends <= insert_n)
                    row = dict(input_policy=input_name, hop_seconds=hop, output_policy=kind,
                               background_mode=mode, episode_id=e["episode_id"],
                               label=e["label"], group_id=e["group_id"],
                               phase_offset_seconds=e.get("phase_offset_seconds", 0.),
                               first_correct_audio_time_from_insert_s=first,
                               correct_by_insert_2s_audio_time=bool(np.any(match & (times <= 2.))),
                               any_wrong_label=bool(np.any(alert & (labels != e["label_id"]))),
                               background_any_alert=bool(alert.any()) if is_background else None,
                               synthetic_prefix_any_alert=bool(decision["accepted"][prefix].any()))
                    local.append(row)
                details.extend(local)
                foreground = [r for r in local if r["label"] != "background"]
                background = [r for r in local if r["label"] == "background"]
                hits = [r["first_correct_audio_time_from_insert_s"] for r in foreground
                        if r["first_correct_audio_time_from_insert_s"] is not None]
                def mean(items):
                    return float(np.mean(items)) if items else None
                summary.append(dict(
                    input_policy=input_name, hop_seconds=hop, output_policy=kind, background_mode=mode,
                    foreground_episodes=len(foreground),
                    foreground_recorded_groups=len({r["group_id"] for r in foreground}),
                    foreground_hit_by_insert_2s_audio_time=mean([r["correct_by_insert_2s_audio_time"] for r in foreground]),
                    foreground_missed_by_insert_2s=sum(not r["correct_by_insert_2s_audio_time"] for r in foreground),
                    first_hit_audio_time_s_p50_detected_only=float(np.median(hits)) if hits else None,
                    first_hit_audio_time_s_p95_detected_only=float(np.quantile(hits, .95)) if hits else None,
                    foreground_episode_any_wrong_label_rate=mean([r["any_wrong_label"] for r in foreground]),
                    background_episode_any_alert_rate=mean([r["background_any_alert"] for r in background]),
                    synthetic_prefix_episode_any_alert_rate=mean([r["synthetic_prefix_any_alert"] for r in local]),
                    scheduled_windows=int(selected.sum()),
                    scheduled_model_calls=int(selected.sum() * len(outputs)),
                    steady_model_calls_per_audio_s=len(outputs) / hop,
                    steady_calls_relative_to_I0=2. / hop,
                    sum_isolated_worker_service_s=float(sum(s[selected].sum() for s in services.values())),
                ))
    for input_name, hop in INPUT_POLICIES.items():
        selected = (ends - window_n) % round(hop * sr) == 0
        for model, service in services.items():
            worker_rows.append(dict(input_policy=input_name, model=model, hop_seconds=hop,
                                    scheduled_calls=int(selected.sum()),
                                    isolated_service_ms_p50=float(np.median(service[selected]) * 1000),
                                    isolated_service_ms_p95=float(np.quantile(service[selected], .95) * 1000),
                                    isolated_service_demand_ratio=float(service[selected].mean() / hop)))
    for row in summary + details:
        row["experiment_id"] = config["selection_spec"]["experiment_id"]
    write_csv(out / "input_selection_comparison.csv", summary)
    write_csv(out / "input_selection_per_episode.csv", details)
    write_csv(out / "input_isolated_worker_cost.csv", worker_rows)
    write_json(out / "input_selection_scope.json", dict(
        fingerprint=fingerprint, annotation=manifest["annotation"], source_scope=config.get("scope"),
        checkpoint_sha256=checkpoint_hashes, total_replayed_audio_seconds=duration,
        limitations=[
            "Clip insertion is not annotated event onset; audio time excludes inference/queue/UI completion.",
            "Repeated insertion phases are correlated; they are not independent recorded groups.",
            "Zero padding is controlled silence, not real ambient negative audio.",
            "Windows use one base-hop cache; this does not measure live policy-specific concurrency, queueing or thermal effects.",
            "S1/S2 cutoffs fit static clips and are not calibrated for continuous rolling windows.",
            "Isolated-worker service demand is not measured utilization on a shared GPU or Raspberry Pi.",
        ]))
    for row in summary:
        print(row)
    print("Saved:", out / "input_selection_comparison.csv")


if __name__ == "__main__":
    main()
