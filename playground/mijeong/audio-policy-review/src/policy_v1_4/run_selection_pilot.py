"""No-voting specialist max pilot on saved predictions; NumPy only.

Default cached project artifacts are an adaptation reference, not verified
checkpoints of the user's best no-voting experiment. Custom heads are supported.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
from policy_core import MODEL_NAMES
from project_io import (ProjectFiles, common_args, load_aligned_predictions,
                        resolve_experiment_spec, write_csv, write_json)
from selection_core import BACKGROUND_MODES, owned_candidates, decide, subset, fit_thresholds


def sha_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_custom(files, spec_path):
    spec_path = Path(spec_path).resolve()
    spec = json.loads(spec_path.read_text(encoding="utf-8-sig"))
    if spec.get("ready_for_evaluation") is not True:
        raise ValueError("Set ready_for_evaluation=true only after verifying actual run paths and head labels")
    if set(spec["prediction_files"]) != set(spec["expert_classes"]):
        raise ValueError("prediction_files and expert_classes models differ")
    canonical, truth, outputs, provenance = None, None, {}, {}
    score_type = spec["score_type"]
    key = "logits" if score_type == "softmax_logits" else "probabilities"
    for model, relative in spec["prediction_files"].items():
        path = Path(relative)
        path = path if path.is_absolute() else files.locate(relative)
        with np.load(path, allow_pickle=False) as data:
            paths = data["paths"].astype(str)
            y = data["y_true"].astype(np.int64)
            values = data[key].astype(np.float64)
        if len(paths) != len(y) or len(values) != len(paths) or len(set(paths.tolist())) != len(paths):
            raise ValueError("invalid or duplicate paths in custom predictions")
        if canonical is None:
            canonical, truth = paths, y
        else:
            index = {p: i for i, p in enumerate(paths)}
            if set(index) != set(canonical.tolist()):
                raise ValueError("custom prediction samples differ")
            order = [index[p] for p in canonical]
            values, y = values[order], y[order]
            if not np.array_equal(y, truth):
                raise ValueError("custom prediction targets differ")
        outputs[model] = values
        provenance[model] = {"path": str(path), "sha256": sha_file(path)}
    # y_true is always an index into the project's GLOBAL class_names.txt,
    # not a local head index. Label names in metadata are independently checked.
    return canonical, truth, outputs, spec, provenance


def measure(candidates, y, kind, common, thresholds):
    d = decide(candidates, kind, common, thresholds)
    bg = candidates.class_names.index("background")
    fg = y != bg
    correct = d["accepted"] & (d["top_id"] == y)
    def rate(a, b):
        return float(a / b) if b else None
    f1s, per_class = [], []
    for i, label in enumerate(candidates.class_names):
        if i == bg:
            continue
        emitted = d["accepted"] & (d["top_id"] == i)
        tp = int((emitted & (y == i)).sum())
        support, emitted_count = int((y == i).sum()), int(emitted.sum())
        f1 = 2 * tp / (support + emitted_count) if support + emitted_count else 0.
        f1s.append(f1)
        per_class.append(dict(policy=kind, label=label, support=support, accepted=emitted_count,
                              precision=rate(tp, emitted_count), recall=rate(tp, support), f1=f1))
    row = dict(policy=kind, clips=len(y), accepted=int(d["accepted"].sum()),
               accepted_precision=rate(correct.sum(), d["accepted"].sum()),
               foreground_correct_accept_rate=rate(correct.sum(), fg.sum()),
               foreground_accept_rate=rate((d["accepted"] & fg).sum(), fg.sum()),
               background_false_accept_rate=rate((d["accepted"] & ~fg).sum(), (~fg).sum()),
               foreground_macro_f1=float(np.mean(f1s)),
               unconfirmed=int((d["status"] == "unconfirmed").sum()),
               background_decisions=int((d["status"] == "background").sum()))
    return row, per_class


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    common_args(parser)
    parser.add_argument("--selection-spec", help="Actual no-voting run paths/head labels, JSON")
    parser.add_argument("--background-mode", choices=(*BACKGROUND_MODES, "all"), default="all",
                        help="all is sensitivity analysis, not three verified baseline implementations")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--target-precision", type=float, default=.95)
    parser.add_argument("--minimum-accepted", type=int, default=20)
    args = parser.parse_args()
    files = ProjectFiles(args.project_root, args.extra_root)
    names = files.class_names
    if args.selection_spec:
        paths, y, outputs, spec, provenance = load_custom(files, args.selection_spec)
        scope = "custom_user_verified_run_spec; original runtime selector not automatically verified"
    else:
        original = resolve_experiment_spec(files, args.assignment, args.ratio)
        paths, y, z = load_aligned_predictions(files, original["specialist_dir"], "val")
        outputs = {m: z[i] for i, m in enumerate(MODEL_NAMES)}
        spec = {"experiment_id": original["experiment_id"], "expert_classes": original["expert_classes"],
                "model_class_names": {m: names for m in MODEL_NAMES}, "score_type": "softmax_logits",
                "checkpoint_files": {m: f"{original['specialist_dir']}/{m}/best."
                                     + ("keras" if m == "yamnet" else "pt") for m in MODEL_NAMES}}
        provenance = {m: {"path": str(files.locate(f"{original['specialist_dir']}/{m}/val_predictions.npz")),
                          "sha256": sha_file(files.locate(f"{original['specialist_dir']}/{m}/val_predictions.npz"))}
                      for m in MODEL_NAMES}
        scope = "supplied_cached_specialists_adaptation_reference; NOT verified best no-voting run"
    groups_by_path = {}
    metadata_path = files.locate("metadata.csv")
    with metadata_path.open(encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            if row["split"] == "val":
                value = (row["group_id"], row["label"])
                if row["path"] in groups_by_path and groups_by_path[row["path"]] != value:
                    raise ValueError("conflicting metadata rows")
                groups_by_path[row["path"]] = value
    groups = []
    for path, target in zip(paths, y):
        if path not in groups_by_path or not 0 <= target < len(names):
            raise ValueError("prediction absent from validation metadata or invalid target")
        group, label = groups_by_path[path]
        if not group or names[int(target)] != label:
            raise ValueError("global target labels or recorded group IDs differ from metadata")
        groups.append(group)
    groups = np.array(groups)
    calibration = np.array([int(hashlib.sha256(f"{args.seed}|{g}".encode()).hexdigest()[:16], 16) % 2 == 0
                            for g in groups])
    evaluation = ~calibration
    overlap = set(groups[calibration]) & set(groups[evaluation])
    if overlap or not calibration.any() or not evaluation.any():
        raise ValueError("invalid recording-group split")
    modes = BACKGROUND_MODES if args.background_mode == "all" else (args.background_mode,)
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    results, by_class, fits = [], [], {}
    for mode in modes:
        candidates = owned_candidates(outputs, names, spec["expert_classes"], spec["model_class_names"],
                                      mode, spec["score_type"])
        common, thresholds = fit_thresholds(subset(candidates, calibration), y[calibration],
                                           args.target_precision, args.minimum_accepted)
        fits[mode] = dict(global_threshold=common, class_thresholds=thresholds)
        for kind in ("S0", "S1", "S2"):
            row, classes = measure(subset(candidates, evaluation), y[evaluation], kind, common, thresholds)
            row.update(background_mode=mode, experiment_id=spec["experiment_id"])
            results.append(row)
            for c in classes:
                c.update(background_mode=mode)
            by_class.extend(classes)
    config = dict(schema="owned_max_selection_v1", scope=scope, selection_spec=spec,
                  prediction_provenance=provenance, metadata_sha256=sha_file(metadata_path),
                  class_names=names, seed=args.seed, policies=fits,
                  target_precision=args.target_precision, minimum_accepted=args.minimum_accepted,
                  calibration_clips=int(calibration.sum()), evaluation_clips=int(evaluation.sum()),
                  calibration_groups=len(set(groups[calibration])), evaluation_groups=len(set(groups[evaluation])),
                  group_overlap=len(overlap), score_renormalization=False, voting=False,
                  calibration_scope="Exploratory held-out part of validation; source training/assignment may use validation. "
                                    "No independent risk guarantee, real-time timing, or event onset measurements.")
    write_json(out / "selection_config.json", config)
    # Preserve the same evaluation groups for rolling-window input experiments.
    np.savez_compressed(out / "selection_samples.npz", paths=paths, y_true=y,
                        group_ids=groups, calibration_mask=calibration, evaluation_mask=evaluation)
    write_csv(out / "selection_comparison.csv", results)
    write_csv(out / "selection_per_class.csv", by_class)
    print(scope)
    for row in results:
        print(row)
    print("Saved:", out)


if __name__ == "__main__":
    main()
