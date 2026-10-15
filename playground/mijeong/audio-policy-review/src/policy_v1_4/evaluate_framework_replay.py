"""Stage B/C/D: input, temporal and bounded asynchronous decision experiments.

Consumes v1.3 GPU exports prepared from run_framework_pilot.py config. Timing
is either audio-clock only or a declared independent-worker FIFO simulation.
Neither timing mode is measured Raspberry Pi/Orin end-to-end latency.
"""
from __future__ import annotations
import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path
import numpy as np
from evaluate_selection_replay import load_exports
from framework_core import (FUSIONS, RULES, AdaptiveHop, TemporalConfirmation,
                            common_probabilities, fusion_candidates, decide_rule,
                            certify_partial, cutoff_for)
from policy_core import INPUT_POLICIES
from project_io import write_csv, write_json


def evaluate_episode(e, indices, ends, candidate, decision, service, sr, base_hop, input_name,
                     timing, temporal_kind, names, rule, fitted, adaptive, temporal_parameters,
                     partial_p=None, partial_w=None, normalize_classes=False, async_policy="wait_all"):
    lookup = {int(ends[i]): i for i in indices}
    first_end, final_end = min(lookup), max(lookup)
    insert = float(e["clip_insert_seconds"])
    window_s = 2.
    worker_finish = np.zeros(service.shape[0])
    engine = TemporalConfirmation(names, temporal_kind, **temporal_parameters)
    cursor = first_end
    trace = []
    actual_calls = 0
    while cursor <= final_end:
        if cursor not in lookup:
            raise ValueError("input scheduler requested a window absent from cache")
        i = lookup[cursor]
        audio_end = cursor / sr
        worker_finish = (np.maximum(worker_finish, audio_end) + service[:, i]
                         if timing == "independent_worker_sim" else np.full(len(worker_finish), audio_end))
        full_ready = float(worker_finish.max())
        now = full_ready
        top, accepted = int(decision["top_id"][i]), bool(decision["accepted"][i])
        ready_count, bound_lo, bound_hi = len(worker_finish), None, None
        visible_scores = candidate.scores[i]
        if async_policy == "bounded":
            arrived = np.zeros(len(worker_finish), dtype=bool)
            visible_p = np.full_like(partial_p[:, i], np.nan)
            for model in np.argsort(worker_finish, kind="stable"):
                arrived[model] = True
                visible_p[model] = partial_p[model, i]
                if arrived.all():
                    break  # Exact full decision handles ties and boundary rounding.
                certified = certify_partial(visible_p, arrived, partial_w, normalize_classes,
                                             names, rule, fitted)
                if certified is not None:
                    if certified["top_id"] != top or certified["accepted"] != accepted:
                        raise AssertionError("certificate differs from full-vote oracle; no result is valid")
                    # The full-vote values above serve only as a post-hoc
                    # compatibility oracle. Use the arrived-data certificate
                    # itself as the policy's early output.
                    top, accepted = int(certified["top_id"]), bool(certified["accepted"])
                    now, ready_count = float(worker_finish[model]), int(arrived.sum())
                    bound_lo, bound_hi = certified["score_lower"], certified["score_upper"]
                    # T0 uses only certified label/acceptance; no unseen score
                    # vector is supplied to a temporal or adaptive policy.
                    visible_scores = np.zeros(len(names))
                    visible_scores[top] = bound_lo
                    break
        if async_policy == "bounded":
            displayed, reason = None, "pending_certificate_event"
        else:
            displayed, reason = engine.step(now, audio_end, visible_scores, top, accepted,
                                             lambda c: cutoff_for(c, names, rule, fitted))
        actual_calls += len(worker_finish)  # Jobs are not cancelled after early display.
        trace.append(dict(audio_end_s=audio_end, decision_time_s=now, result_age_s=now - audio_end,
                          displayed_id=displayed, raw_top_id=top, accepted=accepted,
                          confirmation_reason=reason, models_ready_at_decision=ready_count,
                          ignored_stale_window=False,
                          early_wait_saved_s=full_ready - now,
                          score_lower=bound_lo, score_upper=bound_hi))
        if input_name == "I3":
            hop = adaptive.choose(candidate.scores[i], names, candidate.eligible[i])
            # No next request can depend on a result before its completion.
            target = max(cursor + round(hop * sr), full_ready * sr)
            cursor = int(math.ceil((target - 1e-8) / base_hop)) * base_hop
        else:
            cursor += round(INPUT_POLICIES[input_name] * sr)
    if async_policy == "bounded":
        # Some newer windows certify before an older window waiting for a slow
        # worker. Deliver completion events chronologically and never revert
        # the display to an older input window.
        trace.sort(key=lambda r: (r["decision_time_s"], -r["audio_end_s"]))
        latest_end = -math.inf
        for r in trace:
            if r["audio_end_s"] < latest_end:
                r.update(ignored_stale_window=True, displayed_id=engine.active,
                         confirmation_reason="superseded_window")
                continue
            latest_end = r["audio_end_s"]
            visible = np.zeros(len(names))
            visible[r["raw_top_id"]] = r["score_lower"] or 0.
            displayed, reason = engine.step(r["decision_time_s"], r["audio_end_s"], visible,
                                             r["raw_top_id"], r["accepted"],
                                             lambda c: cutoff_for(c, names, rule, fitted))
            r.update(displayed_id=displayed, confirmation_reason=reason)
    fg = e["label"] != "background"
    evidence = [r for r in trace if not r["ignored_stale_window"] and r["audio_end_s"] > insert
                and r["audio_end_s"] - window_s < insert + window_s]
    correct = [r["decision_time_s"] - insert for r in evidence
               if fg and r["displayed_id"] == e["label_id"]]
    first = min(correct) if correct else None
    previous, changes, switches = None, 0, 0
    for r in trace:
        if r["ignored_stale_window"]:
            continue
        current = r["displayed_id"]
        changes += int(current != previous)
        switches += int(current is not None and previous is not None and current != previous)
        previous = current
    row = dict(episode_id=e["episode_id"], label=e["label"], group_id=e["group_id"],
               phase_offset_seconds=e.get("phase_offset_seconds", 0.),
               first_correct_from_clip_s=first, correct_by_clip_2s=bool(first is not None and first <= 2.),
               any_wrong_label=any(r["displayed_id"] is not None and r["displayed_id"] != e["label_id"] for r in evidence),
               background_any_alert=any(r["displayed_id"] is not None for r in evidence) if not fg else None,
               synthetic_prefix_any_alert=any(r["displayed_id"] is not None and r["audio_end_s"] <= insert for r in trace),
               display_changes=changes, nonempty_label_switches=switches,
               scheduled_windows=len(trace), scheduled_model_calls=actual_calls,
               early_certified_windows=sum(r["models_ready_at_decision"] < len(worker_finish) for r in trace),
               total_early_wait_saved_s=sum(r["early_wait_saved_s"] for r in trace),
               stale_output_rejections=sum(r["confirmation_reason"] == "stale_result" for r in trace),
               superseded_output_windows=sum(r["ignored_stale_window"] for r in trace),
               result_age_s_p95=float(np.quantile([r["result_age_s"] for r in trace], .95)))
    return row, trace


def summarize(rows):
    fg, bg = [r for r in rows if r["label"] != "background"], [r for r in rows if r["label"] == "background"]
    hits = [r["first_correct_from_clip_s"] for r in fg if r["first_correct_from_clip_s"] is not None]
    def avg(values):
        return float(np.mean(values)) if values else None
    return dict(foreground_episodes=len(fg), foreground_groups=len({r["group_id"] for r in fg}),
                hit_by_clip_2s=avg([r["correct_by_clip_2s"] for r in fg]),
                missed_by_clip_2s=sum(not r["correct_by_clip_2s"] for r in fg),
                first_correct_s_p50_detected_only=float(np.median(hits)) if hits else None,
                first_correct_s_p95_detected_only=float(np.quantile(hits, .95)) if hits else None,
                wrong_label_episode_rate=avg([r["any_wrong_label"] for r in fg]),
                background_episode_alert_rate=avg([r["background_any_alert"] for r in bg]),
                display_changes=sum(r["display_changes"] for r in rows),
                nonempty_label_switches=sum(r["nonempty_label_switches"] for r in rows),
                scheduled_windows=sum(r["scheduled_windows"] for r in rows),
                scheduled_model_calls=sum(r["scheduled_model_calls"] for r in rows),
                early_certified_windows=sum(r["early_certified_windows"] for r in rows),
                total_early_wait_saved_s=sum(r["total_early_wait_saved_s"] for r in rows),
                stale_output_rejections=sum(r["stale_output_rejections"] for r in rows),
                superseded_output_windows=sum(r["superseded_output_windows"] for r in rows))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    parser.add_argument("--result-tag", default="framework_replay")
    parser.add_argument("--fusions", choices=FUSIONS, nargs="+")
    parser.add_argument("--rules", choices=RULES, nargs="+", default=["R0"])
    parser.add_argument("--inputs", choices=("I0", "I1", "I2", "I3"), nargs="+", default=["I0", "I1", "I2"])
    parser.add_argument("--temporal", choices=("T0", "T1", "T2"), nargs="+", default=["T0"])
    parser.add_argument("--async-policy", choices=("wait_all", "bounded"), default="wait_all")
    parser.add_argument("--timing", choices=("audio_clock", "independent_worker_sim"), default="audio_clock")
    parser.add_argument("--fast-hop", type=float, default=.5)
    parser.add_argument("--slow-hop", type=float, default=1.)
    parser.add_argument("--adaptive-confidence", type=float, default=.75)
    parser.add_argument("--adaptive-margin", type=float, default=.2)
    parser.add_argument("--urgent-labels", nargs="+", default=["siren", "fire_alarm", "scream", "cry", "car_horn"])
    parser.add_argument("--confirm-seconds", type=float, default=.5)
    parser.add_argument("--transient-labels", nargs="+", default=["gun", "glass_breaking", "knock"])
    parser.add_argument("--strong-score", type=float, default=.9)
    parser.add_argument("--strong-margin", type=float, default=.2)
    parser.add_argument("--max-result-age", type=float, default=2.)
    args = parser.parse_args()
    if not args.result_tag or any(c in args.result_tag for c in ("/", "\\", ":")) or args.result_tag in (".", ".."):
        raise ValueError("result-tag must be a simple output filename prefix")
    if args.async_policy == "bounded" and ("I3" in args.inputs or args.temporal != ["T0"]):
        raise ValueError("Bounded comparison currently supports fixed input hops and T0 only; do not use unseen full scores in adaptive/temporal decisions")
    out = Path(args.out).resolve()
    cfg_bytes, manifest_bytes = (out / "selection_config.json").read_bytes(), (out / "replay_manifest.json").read_bytes()
    cfg, manifest = json.loads(cfg_bytes), json.loads(manifest_bytes)
    if cfg.get("schema") != "policy_framework_v1" or manifest.get("config_file") != "selection_config.json":
        raise ValueError("Use run_framework_pilot.py then prepare_replay.py --selection")
    fw, spec, names = cfg["framework"], cfg["selection_spec"], cfg["class_names"]
    fusions = args.fusions or fw["fusions"]
    if set(fusions) - set(fw["fusions"]):
        raise ValueError("Requested fusion not fitted in framework config")
    if args.async_policy == "bounded" and "F0" in fusions:
        raise ValueError("Partial-vote certificates apply to F1/F2, not owner max F0")
    fingerprint = hashlib.sha256(cfg_bytes + manifest_bytes).hexdigest()
    keys, outputs, services, checkpoint_hashes = load_exports(out, cfg, fingerprint)
    models = fw["model_order"]
    outputs = {m: outputs[m] for m in models}
    service = np.stack([services[m] for m in models])
    p_common = None
    if any(f != "F0" for f in fusions):
        _, p_common = common_probabilities(outputs, names, spec["model_class_names"], spec["score_type"])
    sr, base = int(manifest["sample_rate"]), round(manifest["base_hop_seconds"] * manifest["sample_rate"])
    if manifest["window_seconds"] != 2. or base <= 0:
        raise ValueError("This pilot fixes W=2s and requires a positive cache hop")
    adaptive = AdaptiveHop(args.fast_hop, args.slow_hop, args.adaptive_confidence, args.adaptive_margin, tuple(args.urgent_labels))
    for hop in (*INPUT_POLICIES.values(), args.fast_hop, args.slow_hop):
        if not math.isfinite(hop) or hop <= 0 or round(hop * sr) < base or round(hop * sr) % base:
            raise ValueError("Every input hop must lie on the cached base grid")
    temporal_parameters = dict(confirm_seconds=args.confirm_seconds, strong_score=args.strong_score,
                               strong_margin=args.strong_margin, max_age=args.max_result_age,
                               transient=tuple(args.transient_labels))
    ends = np.array([int(k.split(":")[1]) for k in keys])
    episode_ids = np.array([int(k.split(":")[0]) for k in keys])
    expected = set()
    for e in manifest["episodes"]:
        total = round((e["clip_insert_seconds"] + 2. + manifest["tail_seconds"]) * sr)
        expected.update(f"{e['episode_id']}:{n}" for n in range(2 * sr, total + 1, base))
        if names[e["label_id"]] != e["label"]:
            raise ValueError("episode label mismatch")
    if len({e["episode_id"] for e in manifest["episodes"]}) != len(manifest["episodes"]) or set(keys.tolist()) != expected:
        raise ValueError("incomplete or invalid replay grid")
    summaries, details, traces = [], [], []
    for fusion in fusions:
        candidate = fusion_candidates(outputs, names, spec["expert_classes"], spec["model_class_names"], fusion,
                                      spec["score_type"], fw["classwise_weights"], fw["owner_background"])
        fitted = fw["acceptance"][fusion]
        partial_w = (np.full((len(models), len(names)), 1. / len(models)) if fusion == "F1"
                     else np.asarray(fw["classwise_weights"]))
        for rule, input_name, temporal_kind in itertools.product(args.rules, args.inputs, args.temporal):
            decision = decide_rule(candidate, rule, fitted)
            setting = dict(fusion=fusion, rule=rule, input_policy=input_name, temporal=temporal_kind,
                           timing=args.timing, async_policy=args.async_policy)
            local = []
            for e in manifest["episodes"]:
                row, trace = evaluate_episode(e, np.flatnonzero(episode_ids == e["episode_id"]), ends,
                    candidate, decision, service, sr, base, input_name, args.timing, temporal_kind, names,
                    rule, fitted, adaptive, temporal_parameters, p_common, partial_w, fusion == "F2", args.async_policy)
                local.append(dict(row, **setting))
                traces.extend(dict(r, episode_id=e["episode_id"], **setting) for r in trace)
            details.extend(local)
            summaries.append(dict(summarize(local), **setting))
    write_csv(out / f"{args.result_tag}_comparison.csv", summaries)
    write_csv(out / f"{args.result_tag}_per_episode.csv", details)
    write_csv(out / f"{args.result_tag}_trace.csv", traces)
    write_json(out / f"{args.result_tag}_scope.json", dict(
        arguments=vars(args), fingerprint=fingerprint, checkpoint_sha256=checkpoint_hashes,
        source_scope=cfg.get("scope"), model_jobs_cancelled=False,
        limitations=["Audio-clock values exclude processing. Independent-worker FIFO timing is a SIMULATION from isolated timings, not measured shared-device latency.",
                     "Clip insertion is not actual event onset. Repeated phases are correlated. No formal risk guarantee.",
                     "Static clip thresholds transferred to rolling windows are not calibrated for streaming false-alert rates.",
                     "All model jobs execute even when a bounded decision is early. No inference FLOP savings are claimed.",
                     "Adaptive input uses only completed selected-window outputs, never skipped/future predictions.",
                     "Actual Bluetooth transport is outside scope; actual concurrent workers/UI timing require device validation."]))
    for row in summaries:
        print(row)


if __name__ == "__main__":
    main()
