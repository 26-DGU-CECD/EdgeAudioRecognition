from __future__ import annotations
import csv, hashlib, json
from pathlib import Path
import numpy as np
from policy_core import MODEL_NAMES, fuse_logits


class ProjectFiles:
    def __init__(self, project_root, extra_roots=()):
        self.root = Path(project_root).expanduser().resolve()
        self.roots = [self.root] + [Path(p).expanduser().resolve() for p in extra_roots]
        if not self.root.is_dir():
            raise FileNotFoundError(self.root)

    def locate(self, relative):
        candidates = [root / relative for root in self.roots if (root / relative).is_file()]
        if not candidates:
            raise FileNotFoundError(f"{relative}: not found under {self.roots}")
        if len(candidates) > 1:
            hashes = {hashlib.sha256(p.read_bytes()).hexdigest() for p in candidates}
            if len(hashes) != 1:
                raise ValueError(f"Conflicting source copies: {relative}")
        return candidates[0]

    @property
    def class_names(self):
        return [s.strip() for s in self.locate("class_names.txt").read_text(
            encoding="utf-8-sig").splitlines() if s.strip()]


def experiment_paths(assignment="predefined", ratio=0.70):
    if not 0 < ratio <= 1:
        raise ValueError("ratio must be in (0,1]")
    tag = f"nonexpert_{int(round(ratio * 100)):03d}"
    if assignment == "predefined":
        return f"runs/E_predefined_specialists/{tag}", f"runs/F_predefined_weighted_ensemble/{tag}"
    if assignment == "automatic":
        return f"runs/G_auto_specialists/{tag}", f"runs/G_auto_weighted_ensemble/{tag}"
    raise ValueError(assignment)


def read_weights(files, ensemble_dir, class_names):
    rows = list(csv.DictReader(files.locate(
        f"{ensemble_dir}/classwise_model_weights.csv").open(
            encoding="utf-8-sig", newline="")))
    if [row[next(iter(row))] for row in rows] != class_names:
        raise ValueError("Weight class order does not match class_names.txt")
    return np.array([[float(row[m]) for row in rows] for m in MODEL_NAMES])


def load_aligned_predictions(files, specialist_dir, split="val"):
    canonical, truth, logits = None, None, []
    for model in MODEL_NAMES:
        with np.load(files.locate(f"{specialist_dir}/{model}/{split}_predictions.npz"),
                     allow_pickle=False) as data:
            paths = data["paths"].astype(str)
            y = data["y_true"].astype(np.int64)
            z = data["logits"].astype(np.float64)
        if len(set(paths.tolist())) != len(paths):
            raise ValueError("Duplicate prediction paths")
        if canonical is None:
            canonical, truth = paths, y
        else:
            index = {p: i for i, p in enumerate(paths)}
            if set(index) != set(canonical.tolist()):
                raise ValueError("Prediction path mismatch")
            order = np.array([index[p] for p in canonical])
            y, z = y[order], z[order]
            if not np.array_equal(y, truth):
                raise ValueError("Targets do not align")
        logits.append(z)
    return canonical, truth, np.stack(logits)


def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False),
                          encoding="utf-8")


def write_csv(path, rows):
    if not rows:
        return
    with Path(path).open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def common_args(parser):
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--extra-root", action="append", default=[])
    parser.add_argument("--assignment", choices=["predefined", "automatic"], default="predefined")
    parser.add_argument("--ratio", type=float, default=0.70)
    parser.add_argument("--out", default="pilot_results")



def validate_experiment_spec(spec, class_names):
    """Declarative paths and ownership only; never execute code from a spec."""
    if spec.get("ready_for_evaluation") is False:
        raise ValueError("Template experiment is not ready; validate assignment and train models first")
    required = {"experiment_id", "assignment_type", "nonexpert_fraction",
                "specialist_dir", "ensemble_dir", "expert_classes"}
    missing = required - set(spec)
    if missing:
        raise ValueError(f"Experiment spec missing fields: {sorted(missing)}")
    if not isinstance(spec["experiment_id"], str) or not spec["experiment_id"].strip():
        raise ValueError("experiment_id must be a nonempty string")
    if not 0 < float(spec["nonexpert_fraction"]) <= 1:
        raise ValueError("nonexpert_fraction must be in (0,1]")
    for key in ("specialist_dir", "ensemble_dir"):
        value = str(spec[key])
        parts = Path(value).parts
        if not value or Path(value).is_absolute() or ".." in value.replace("\\", "/").split("/") or ":" in value:
            raise ValueError(f"{key} must be a relative path inside the project")
    experts = spec["expert_classes"]
    if set(experts) != set(MODEL_NAMES):
        raise ValueError("expert_classes must specify all three model names")
    flat = []
    for model in MODEL_NAMES:
        if not isinstance(experts[model], list) or not experts[model]:
            raise ValueError("each model must have at least one expert label")
        flat.extend(experts[model])
    foreground = set(class_names) - {"background"}
    if len(flat) != len(set(flat)) or set(flat) != foreground:
        raise ValueError("each foreground label must have exactly one owner; exclude background")
    return spec


def resolve_experiment_spec(files, assignment="predefined", ratio=0.70, spec_path=None):
    if spec_path:
        spec = json.loads(Path(spec_path).read_text(encoding="utf-8-sig"))
    else:
        specialist_dir, ensemble_dir = experiment_paths(assignment, ratio)
        tag = f"nonexpert_{int(round(ratio * 100)):03d}"
        folder = "predefined" if assignment == "predefined" else "automatic"
        source = files.locate(
            f"specialist_metadata/{folder}/{tag}/specialist_assignment.json")
        assignment_data = json.loads(source.read_text(encoding="utf-8-sig"))
        spec = {
            "experiment_id": f"{'R0' if assignment == 'predefined' else 'R1'}_{folder}_{tag}",
            "assignment_type": assignment, "nonexpert_fraction": float(ratio),
            "specialist_dir": specialist_dir, "ensemble_dir": ensemble_dir,
            "expert_classes": assignment_data["expert_classes"],
            "selection_basis": "predefined" if assignment == "predefined"
                               else "baseline_validation_per_class_f1",
            "assignment_file": str(source),
            "assignment_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        }
    validate_experiment_spec(spec, files.class_names)
    return spec


def config_experiment_paths(config):
    """Old v1 configs work; v1.1 uses recorded experiment directories."""
    spec = config.get("experiment_spec")
    if spec:
        validate_experiment_spec(spec, config["class_names"])
        return spec["specialist_dir"], spec["ensemble_dir"]
    return experiment_paths(config["assignment"], config["ratio"])


def read_transfer_policy(path, class_names, seed):
    config = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if config["class_names"] != list(class_names):
        raise ValueError("Transfer requires identical label identities and class order")
    if int(config["seed"]) != int(seed):
        raise ValueError("Use the same --seed as the source policy for matched group evaluation")
    expected = set(class_names) - {"background"}
    if set(config["class_thresholds"]) != expected:
        raise ValueError("Transferred thresholds do not cover the same foreground labels")
    # Transfer only policy parameters. Never import old model/fusion weights.
    keys = ("global_threshold", "class_thresholds", "target_precision",
            "minimum_accepted", "seed", "evaluation_set_sha256")
    return {key: config[key] for key in keys if key in config}
