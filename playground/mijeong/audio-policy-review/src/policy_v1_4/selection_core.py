"""No-voting specialist selection; native head scores retain their scale.

This module does not average model outputs or normalize across specialists.
Outputs for different input windows must never be compared with each other.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from policy_core import softmax

BACKGROUND_MODES = ("global_max", "exclude", "per_model_gate")


@dataclass(frozen=True)
class CandidateScores:
    scores: np.ndarray
    eligible: np.ndarray
    class_names: tuple[str, ...]

    def top(self):
        if (self.scores.ndim != 2 or self.eligible.shape != self.scores.shape
                or self.scores.shape[1] != len(self.class_names)):
            raise ValueError("invalid candidate dimensions")
        if (not np.isfinite(self.scores).all() or (self.scores < 0).any()
                or (self.scores > 1).any() or not self.eligible.any(axis=1).all()):
            raise ValueError("invalid scores or no eligible candidate")
        # Tie rule is global class order, independent of arrival/model order.
        top = np.where(self.eligible, self.scores, -np.inf).argmax(axis=1)
        return top, self.scores[np.arange(len(top)), top]


def owned_candidates(outputs, class_names, expert_classes, model_class_names,
                     background_mode, score_type="softmax_logits"):
    """Inputs: dict model->[N, actual_head_classes]; different heads supported.

    Apply softmax over each complete native head BEFORE masking nonexpert labels.
    `probabilities` also permits supplied sigmoid scores; neither input form
    guarantees comparability between models trained on different label spaces.
    Background variants are explicit hypotheses, not claimed original behavior.
    """
    names = tuple(class_names)
    if len(names) != len(set(names)) or "background" not in names:
        raise ValueError("unique class names including background required")
    if background_mode not in BACKGROUND_MODES:
        raise ValueError("explicit background mode required")
    if score_type not in ("softmax_logits", "probabilities"):
        raise ValueError("unsupported score_type")
    if not outputs or set(outputs) != set(expert_classes) or set(outputs) != set(model_class_names):
        raise ValueError("model dictionaries must match")
    flat = [c for m in outputs for c in expert_classes[m]]
    if (len(flat) != len(set(flat)) or set(flat) != set(names) - {"background"}
            or any(not expert_classes[m] for m in outputs)):
        raise ValueError("each foreground label must have exactly one owner")
    count = None
    prepared = {}
    for model, values in outputs.items():
        head = tuple(model_class_names[model])
        if len(head) != len(set(head)) or not set(head) <= set(names):
            raise ValueError("invalid model head labels")
        if not set(expert_classes[model]) <= set(head):
            raise ValueError("expert label missing from owner's head")
        if background_mode != "exclude" and "background" not in head:
            raise ValueError("background score needed in every head for this mode")
        a = np.asarray(values, dtype=np.float64)
        if a.ndim != 2 or a.shape[1] != len(head) or not np.isfinite(a).all():
            raise ValueError("invalid head values")
        if count is None:
            count = len(a)
        elif len(a) != count:
            raise ValueError("different window counts")
        p = softmax(a) if score_type == "softmax_logits" else a
        if (p < 0).any() or (p > 1).any():
            raise ValueError("probabilities must be in [0,1]")
        prepared[model] = (head, p)
    scores = np.zeros((count, len(names)), dtype=np.float64)
    eligible = np.zeros_like(scores, dtype=bool)
    backgrounds = []
    for model, (head, p) in prepared.items():
        owned = expert_classes[model]
        local_ids = [head.index(c) for c in owned]
        global_ids = [names.index(c) for c in owned]
        scores[:, global_ids] = p[:, local_ids]
        gate = np.ones(count, dtype=bool)
        if "background" in head:
            bg = p[:, head.index("background")]
            backgrounds.append(bg)
            if background_mode == "per_model_gate":
                gate = p[:, local_ids].max(axis=1) > bg
        eligible[:, global_ids] = gate[:, None]
    bg_id = names.index("background")
    if backgrounds:
        scores[:, bg_id] = np.max(np.stack(backgrounds), axis=0)
    if background_mode == "global_max":
        eligible[:, bg_id] = True
    elif background_mode == "per_model_gate":
        eligible[:, bg_id] = ~eligible.any(axis=1)
    return CandidateScores(scores, eligible, names)


def decide(candidates, kind="S0", global_threshold=None, class_thresholds=None):
    """S0 native max; S1 common reject; S2 label-dependent reject.

    S1/S2 NEVER fall back to a lower-ranked label. None means reject, not 0.
    Rejection is `unconfirmed`, distinct from a model background decision.
    """
    if kind not in ("S0", "S1", "S2"):
        raise ValueError(kind)
    top, score = candidates.top()
    bg = candidates.class_names.index("background")
    if kind == "S0":
        cutoff = np.zeros(len(top))
    elif kind == "S1":
        if global_threshold is not None and not 0 <= global_threshold <= 1:
            raise ValueError("invalid global threshold")
        cutoff = np.full(len(top), np.inf if global_threshold is None else global_threshold)
    else:
        expected = set(candidates.class_names) - {"background"}
        if class_thresholds is None or set(class_thresholds) != expected:
            raise ValueError("S2 needs every foreground threshold")
        for v in class_thresholds.values():
            if v is not None and not 0 <= v <= 1:
                raise ValueError("invalid class threshold")
        cutoff = np.array([
            np.inf if class_thresholds.get(candidates.class_names[i]) is None
            else class_thresholds[candidates.class_names[i]] for i in top])
    accepted = (top != bg) & (score >= cutoff)
    status = np.where(top == bg, "background", np.where(accepted, "accepted", "unconfirmed"))
    return dict(top_id=top, score=score, accepted=accepted, status=status)


def subset(candidates, mask):
    return CandidateScores(candidates.scores[mask], candidates.eligible[mask], candidates.class_names)


def fit_thresholds(candidates, y_true, target_precision=.95, minimum_accepted=20):
    """Development-set grid search; no independent risk-control guarantee."""
    y = np.asarray(y_true, dtype=np.int64)
    top, score = candidates.top()
    if (y.shape != top.shape or not 0 < target_precision <= 1 or minimum_accepted < 1
            or (y < 0).any() or (y >= len(candidates.class_names)).any()):
        raise ValueError("invalid fitting data or requirements")
    bg = candidates.class_names.index("background")
    grid = np.unique(np.r_[np.arange(0., 1., .005), .999, 1.])

    def fit(mask):
        best = None
        for tau in grid:
            chosen = mask & (score >= tau)
            n = int(chosen.sum())
            tp = int((chosen & (top == y)).sum())
            if n >= minimum_accepted and tp / n >= target_precision:
                candidate = (tp, -float(tau), float(tau), n, tp / n)
                if best is None or candidate > best:
                    best = candidate
        return None if best is None else best[2]

    common = fit(top != bg)
    by_class = {name: fit(top == i) for i, name in enumerate(candidates.class_names) if i != bg}
    return common, by_class
