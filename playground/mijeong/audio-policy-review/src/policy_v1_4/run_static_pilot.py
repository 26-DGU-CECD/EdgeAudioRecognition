"""Output pilot on existing validation predictions; no Torch/TF imports."""
from __future__ import annotations
import argparse, csv, hashlib
from collections import defaultdict
from pathlib import Path
import numpy as np
from policy_core import OutputPolicy, fit_class_thresholds, fuse_logits
from project_io import (ProjectFiles, common_args, experiment_paths,
                        load_aligned_predictions, read_weights, write_json, write_csv,
                        resolve_experiment_spec, read_transfer_policy)


def binary_fraction(numerator, denominator):
    return float(numerator / denominator) if denominator else None


def metrics(policy, p, y, background_id):
    d = policy.decide(p)
    accepted = d["accepted"]
    correct = accepted & (d["top_id"] == y)
    foreground = y != background_id
    return {
        "policy": policy.kind, "evaluation_clips": int(len(y)),
        "accepted": int(accepted.sum()),
        "accepted_precision": binary_fraction(correct.sum(), accepted.sum()),
        "foreground_correct_accept_rate": binary_fraction(correct.sum(), foreground.sum()),
        "foreground_accept_rate": binary_fraction((accepted & foreground).sum(), foreground.sum()),
        "background_false_accept_rate": binary_fraction(
            (accepted & ~foreground).sum(), (~foreground).sum()),
        "rejected_foreground": int((foreground & ~accepted).sum()),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    common_args(parser)
    parser.add_argument("--experiment-spec", default=None,
                        help="Declarative custom label ownership and model-run paths.")
    parser.add_argument("--policy-config-from", default=None,
                        help="Freeze output thresholds from a previous experiment; keep new model weights.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--global-threshold", type=float, default=0.6)
    parser.add_argument("--target-precision", type=float, default=0.95)
    parser.add_argument("--minimum-accepted", type=int, default=20)
    parser.add_argument("--verify-saved-test-fusion", action="store_true",
                        help="Compatibility check only; test does not fit policy values.")
    args = parser.parse_args()
    files = ProjectFiles(args.project_root, args.extra_root)
    names = files.class_names
    spec = resolve_experiment_spec(
        files, args.assignment, args.ratio, args.experiment_spec)
    specialist_dir, ensemble_dir = spec["specialist_dir"], spec["ensemble_dir"]
    weights = read_weights(files, ensemble_dir, names)
    paths, y, z = load_aligned_predictions(files, specialist_dir, "val")
    p = fuse_logits(z, weights)
    metadata = {}
    with files.locate("metadata.csv").open(encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            if row["split"] == "val":
                path = row["path"]
                if path in metadata and metadata[path] != row["group_id"]:
                    raise ValueError("Conflicting groups")
                metadata[path] = row["group_id"]
    missing = [path for path in paths if path not in metadata]
    if missing:
        raise ValueError(f"{len(missing)} prediction paths absent from validation metadata")
    groups = np.array([metadata[path] for path in paths], dtype=str)
    # Same recording group always goes to the same side, across all labels.
    calibration = np.array([
        int(hashlib.sha256(f"{args.seed}|{group}".encode()).hexdigest()[:16], 16) % 2 == 0
        for group in groups
    ])
    evaluation = ~calibration
    if not calibration.any() or not evaluation.any():
        raise ValueError("Empty group split")
    overlap = set(groups[calibration]) & set(groups[evaluation])
    assert not overlap
    evaluation_set_sha256 = hashlib.sha256("\n".join(sorted(
        f"{path}\t{group}\t{int(target)}"
        for path, group, target in zip(paths[evaluation], groups[evaluation], y[evaluation])
    )).encode()).hexdigest()
    transfer = None
    global_threshold = args.global_threshold
    target_precision, minimum_accepted = args.target_precision, args.minimum_accepted
    if args.policy_config_from:
        transfer = read_transfer_policy(args.policy_config_from, names, args.seed)
        if (transfer.get("evaluation_set_sha256") is not None
                and transfer["evaluation_set_sha256"] != evaluation_set_sha256):
            raise ValueError("Transferred policy evaluation samples differ; use matched data")
        thresholds = transfer["class_thresholds"]
        global_threshold = float(transfer["global_threshold"])
        target_precision = float(transfer["target_precision"])
        minimum_accepted = int(transfer["minimum_accepted"])
        threshold_rows = [
            {"label": label, "threshold": value, "origin": "frozen_transfer"}
            for label, value in thresholds.items()
        ]
    else:
        thresholds, threshold_rows = fit_class_thresholds(
            p[calibration], y[calibration], names,
            target_precision, minimum_accepted,
        )
    policy_parameter_mode = "frozen_transfer" if transfer else "fit_current_scores"
    policies = [
        OutputPolicy(names, "O0"),
        OutputPolicy(names, "O1", global_threshold=global_threshold),
        OutputPolicy(names, "O2", class_thresholds=thresholds),
    ]
    results = [metrics(policy, p[evaluation], y[evaluation], names.index("background"))
               for policy in policies]
    for row in results:
        row.update(experiment_id=spec["experiment_id"],
                   policy_parameter_mode=policy_parameter_mode)
    per_class = []
    for policy in policies:
        d = policy.decide(p[evaluation])
        truth = y[evaluation]
        for cid, label in enumerate(names):
            if label == "background":
                continue
            emitted = d["accepted"] & (d["top_id"] == cid)
            tp = emitted & (truth == cid)
            per_class.append({
                "policy": policy.kind, "label": label,
                "true_clips": int((truth == cid).sum()),
                "true_groups": len(set(groups[evaluation][truth == cid])),
                "accepted": int(emitted.sum()),
                "precision": binary_fraction(tp.sum(), emitted.sum()),
                "correct_accept_rate": binary_fraction(tp.sum(), (truth == cid).sum()),
            })
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    config = {
        "class_names": names, "assignment": spec["assignment_type"],
        "ratio": float(spec["nonexpert_fraction"]),
        "experiment_spec": spec,
        "evaluation_set_sha256": evaluation_set_sha256,
        "policy_parameter_mode": policy_parameter_mode,
        "transferred_policy_file": args.policy_config_from,
        "weights_sha256": hashlib.sha256(np.ascontiguousarray(weights).tobytes()).hexdigest(),
        "global_threshold": global_threshold,
        "class_thresholds": thresholds,
        "target_precision": target_precision, "minimum_accepted": minimum_accepted,
        "seed": args.seed, "weights": weights.tolist(),
        "calibration_clips": int(calibration.sum()),
        "evaluation_clips": int(evaluation.sum()),
        "calibration_groups": len(set(groups[calibration])),
        "evaluation_groups": len(set(groups[evaluation])),
        "group_overlap": len(overlap),
        "scope": "Exploratory validation split; original models/weights used validation. "
                 "No independent risk guarantee or event latency measurement.",
    }
    write_json(out / "policy_config.json", config)
    write_csv(out / "static_policy_comparison.csv", results)
    write_csv(out / "static_per_class.csv", per_class)
    write_csv(out / "threshold_fit.csv", threshold_rows)
    np.savez_compressed(out / "static_scores.npz",
                        probabilities=p, y_true=y, paths=paths, group_ids=groups,
                        calibration_mask=calibration, evaluation_mask=evaluation)
    checks = {"group_overlap": len(overlap), "probability_sum_max_error":
              float(np.max(np.abs(p.sum(axis=1) - 1)))}
    if args.verify_saved_test_fusion:
        test_paths, test_y, test_z = load_aligned_predictions(files, specialist_dir, "test")
        rebuilt = fuse_logits(test_z, weights)
        with np.load(files.locate(f"{ensemble_dir}/test_predictions.npz"),
                     allow_pickle=False) as saved:
            index = {path: i for i, path in enumerate(saved["paths"].astype(str))}
            order = np.array([index[path] for path in test_paths])
            reference = saved["probabilities"][order]
            assert np.array_equal(test_y, saved["y_true"][order])
        checks["test_fusion_max_abs_error"] = float(np.max(np.abs(reference - rebuilt)))
        checks["test_fusion_top1_match"] = float(
            (reference.argmax(axis=1) == rebuilt.argmax(axis=1)).mean())
        if not np.allclose(reference, rebuilt, atol=1e-10, rtol=1e-10):
            raise AssertionError(checks)
    write_json(out / "compatibility_checks.json", checks)
    for row in results:
        print(row)
    print("Saved:", out)


if __name__ == "__main__":
    main()

