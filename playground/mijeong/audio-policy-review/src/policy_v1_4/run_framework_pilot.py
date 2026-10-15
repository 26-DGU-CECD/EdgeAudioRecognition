"""Stage A: fixed W2/H2, compare fusion F0/F1/F2 x acceptance R0/R1/R2."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
from project_io import (ProjectFiles, common_args, resolve_experiment_spec,
                        load_aligned_predictions, write_json, write_csv)
from run_selection_pilot import load_custom, sha_file, measure
from policy_core import MODEL_NAMES
from selection_core import subset
from framework_core import (FUSIONS, RULES, common_probabilities, fit_class_weights,
                            fusion_candidates, fit_acceptance)


def load_data(files, args):
    if args.selection_spec:
        paths, y, outputs, spec, provenance = load_custom(files, args.selection_spec)
        scope = "custom_run_spec; actual historical runtime decision path not independently verified"
    else:
        original = resolve_experiment_spec(files, args.assignment, args.ratio)
        paths, y, z = load_aligned_predictions(files, original["specialist_dir"], "val")
        names = files.class_names
        spec = dict(experiment_id=original["experiment_id"], expert_classes=original["expert_classes"],
                    model_class_names={m: names for m in MODEL_NAMES}, score_type="softmax_logits",
                    checkpoint_files={m: f"{original['specialist_dir']}/{m}/best."
                                      + ("keras" if m == "yamnet" else "pt") for m in MODEL_NAMES})
        outputs = {m: z[i] for i, m in enumerate(MODEL_NAMES)}
        provenance = {m: dict(path=str(files.locate(f"{original['specialist_dir']}/{m}/val_predictions.npz")),
                              sha256=sha_file(files.locate(f"{original['specialist_dir']}/{m}/val_predictions.npz")))
                      for m in MODEL_NAMES}
        scope = "supplied_specialist_cache_reference; NOT verified best historical run"
    lookup = {}
    with files.locate("metadata.csv").open(encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            if row["split"] == "val":
                pair = (row["group_id"], row["label"])
                if row["path"] in lookup and lookup[row["path"]] != pair:
                    raise ValueError("conflicting validation metadata")
                lookup[row["path"]] = pair
    groups = []
    names = files.class_names
    for p, target in zip(paths, y):
        if p not in lookup or not 0 <= target < len(names):
            raise ValueError("prediction missing from validation metadata")
        g, label = lookup[p]
        if not g or names[int(target)] != label:
            raise ValueError("global y_true/group differs from metadata")
        groups.append(g)
    return paths, y, outputs, spec, provenance, np.array(groups), scope


def group_split(groups, seed):
    # Preserve previous seed's evaluation half. Split calibration half again
    # so F2 weights and acceptance thresholds never fit the same group.
    calibration = np.array([int(hashlib.sha256(f"{seed}|{g}".encode()).hexdigest()[:16], 16) % 2 == 0 for g in groups])
    secondary = np.array([int(hashlib.sha256(f"fusion|{seed}|{g}".encode()).hexdigest()[:16], 16) % 2 == 0 for g in groups])
    masks = {"fusion_fit": calibration & secondary, "policy_fit": calibration & ~secondary,
             "evaluation": ~calibration}
    sets = {k: set(groups[v]) for k, v in masks.items()}
    if any(not v.any() for v in masks.values()) or any(sets[a] & sets[b] for a in sets for b in sets if a != b):
        raise ValueError("empty or overlapping three-way group split")
    return masks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    common_args(parser)
    parser.add_argument("--selection-spec")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fusions", choices=FUSIONS, nargs="+", default=list(FUSIONS))
    parser.add_argument("--owner-background", choices=("global_max", "exclude", "per_model_gate"), default="global_max")
    parser.add_argument("--target-precision", type=float, default=.95)
    parser.add_argument("--minimum-accepted", type=int, default=20)
    parser.add_argument("--recall-floor", type=float, default=.5)
    parser.add_argument("--label-recall-floors", help="JSON mapping of foreground labels to recall-floor overrides")
    args = parser.parse_args()
    label_floors = (json.loads(Path(args.label_recall_floors).read_text(encoding="utf-8-sig"))
                    if args.label_recall_floors else {})
    files = ProjectFiles(args.project_root, args.extra_root)
    names = files.class_names
    paths, y, outputs, spec, provenance, groups, scope = load_data(files, args)
    masks = group_split(groups, args.seed)
    model_order = list(outputs)
    weights = None
    if any(f != "F0" for f in args.fusions):
        model_order, p = common_probabilities(outputs, names, spec["model_class_names"], spec["score_type"])
        if "F2" in args.fusions:
            weights = fit_class_weights(p[:, masks["fusion_fit"]], y[masks["fusion_fit"]], names,
                                        model_order, spec["expert_classes"])
    fitted_by_fusion, results, classes, diagnostics = {}, [], [], []
    for fusion in dict.fromkeys(args.fusions):
        candidates = fusion_candidates(outputs, names, spec["expert_classes"], spec["model_class_names"],
                                       fusion, spec["score_type"], weights, args.owner_background)
        fitted = fit_acceptance(subset(candidates, masks["policy_fit"]), y[masks["policy_fit"]],
                                args.target_precision, args.minimum_accepted, args.recall_floor, label_floors)
        fitted_by_fusion[fusion] = fitted
        for row in fitted["diagnostics"]:
            diagnostics.append(dict(fusion=fusion, **row))
        for rule in RULES:
            row, label_rows = measure(subset(candidates, masks["evaluation"]), y[masks["evaluation"]],
                                      {"R0": "S0", "R1": "S1", "R2": "S2"}[rule],
                                      fitted["global_threshold"], fitted["class_thresholds"])
            row.update(fusion=fusion, policy=rule, seed=args.seed,
                       below_recall_floor_labels="|".join(r["label"] for r in label_rows
                                                         if r["recall"] is not None and r["recall"] < label_floors.get(r["label"], args.recall_floor)),
                       global_fit_feasible=fitted["global_fit_feasible"])
            results.append(row)
            for r in label_rows:
                classes.append(dict(r, policy=rule, fusion=fusion))
    # Compatible with v1.3 manifest/GPU exporter. Only F0 baseline is exposed to
    # the old evaluator; the full framework is consumed by the NEW evaluator.
    baseline = fitted_by_fusion.get("F0")
    policy_compat = {} if baseline is None else {args.owner_background: {
        "global_threshold": baseline["global_threshold"], "class_thresholds": baseline["class_thresholds"]}}
    config = dict(schema="policy_framework_v1", scope=scope, selection_spec=spec,
                  prediction_provenance=provenance, class_names=names, seed=args.seed,
                  policies=policy_compat,
                  framework=dict(model_order=list(model_order), fusions=list(fitted_by_fusion),
                                 classwise_weights=None if weights is None else weights.tolist(),
                                 weight_parameters=dict(floor=.05, power=2., expert_boost=1.25),
                                 owner_background=args.owner_background,
                                 acceptance=fitted_by_fusion,
                                 group_split={k: dict(clips=int(v.sum()), groups=len(set(groups[v]))) for k, v in masks.items()},
                                 group_overlap=0),
                  calibration_scope="Exploratory validation re-split. Original checkpoint/model/ownership selection may already have used validation. No independent test/risk guarantee.")
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "selection_config.json", config)
    np.savez_compressed(out / "selection_samples.npz", paths=paths, y_true=y, group_ids=groups,
                        calibration_mask=masks["fusion_fit"] | masks["policy_fit"],
                        evaluation_mask=masks["evaluation"], fusion_fit_mask=masks["fusion_fit"],
                        policy_fit_mask=masks["policy_fit"])
    write_csv(out / "framework_static_comparison.csv", results)
    write_csv(out / "framework_static_per_class.csv", classes)
    write_csv(out / "framework_threshold_diagnostics.csv", diagnostics)
    for r in results:
        print(r)
    print("Saved:", out)


if __name__ == "__main__":
    main()
