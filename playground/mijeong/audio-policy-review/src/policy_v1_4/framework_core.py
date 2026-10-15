"""Deterministic fusion, acceptance, input and confirmation policies.

No policy neural network, internal model exits or model cancellation. Validation
weight/threshold fitting is exploratory and does not provide risk guarantees.
"""
from __future__ import annotations
from dataclasses import dataclass
import math
import numpy as np
from policy_core import softmax, AudioWindow
from selection_core import CandidateScores, owned_candidates, fit_thresholds, decide

FUSIONS = ("F0", "F1", "F2")
RULES = ("R0", "R1", "R2")


def common_probabilities(outputs, names, heads, score_type):
    """Return [models,N,C]. Never zero-fill labels missing from native heads."""
    models = tuple(outputs)
    arrays = []
    for model in models:
        head = list(heads[model])
        if len(head) != len(set(head)) or set(head) != set(names):
            raise ValueError("F1/F2 require every head to cover the SAME complete global label set; do not zero-pad dedicated subset heads")
        a = np.asarray(outputs[model], dtype=np.float64)
        if a.ndim != 2 or a.shape[1] != len(head):
            raise ValueError("head dimensions differ")
        if score_type == "softmax_logits":
            p = softmax(a)
        elif score_type == "probabilities":
            p = a
        else:
            raise ValueError("unknown score type")
        if (not np.isfinite(p).all() or (p < 0).any() or (p > 1).any()
                or not np.allclose(p.sum(axis=1), 1., atol=1e-7)):
            raise ValueError("Voting requires native categorical probabilities; unnormalized/sigmoid heads need a separately validated fusion")
        arrays.append(p[:, [head.index(c) for c in names]])
    return models, np.stack(arrays)


def fit_class_weights(p, y, names, models, experts, floor=.05, power=2., boost=1.25):
    """Existing project formula, fitted only on fusion-fit recording groups."""
    p, y = np.asarray(p), np.asarray(y, dtype=np.int64)
    if p.ndim != 3 or p.shape[0] != len(models) or p.shape[1] != len(y) or p.shape[2] != len(names) or not len(y):
        raise ValueError("invalid fusion-fit data")
    if floor <= 0 or power <= 0 or boost <= 0:
        raise ValueError("positive weight hyperparameters required")
    raw = np.zeros((len(models), len(names)))
    for m, model in enumerate(models):
        pred = p[m].argmax(axis=1)
        for c, label in enumerate(names):
            tp = ((pred == c) & (y == c)).sum()
            den = (pred == c).sum() + (y == c).sum()
            f1 = float(2 * tp / den) if den else 0.
            raw[m, c] = max(f1, floor) ** power * (boost if label in experts[model] else 1.)
    return raw / raw.sum(axis=0, keepdims=True)


def fusion_candidates(outputs, names, experts, heads, kind, score_type="softmax_logits",
                      weights=None, owner_background="global_max"):
    if kind == "F0":
        return owned_candidates(outputs, names, experts, heads, owner_background, score_type)
    if kind not in ("F1", "F2"):
        raise ValueError(kind)
    models, p = common_probabilities(outputs, names, heads, score_type)
    if kind == "F1":
        combined = p.mean(axis=0)
    else:
        w = np.asarray(weights, dtype=float)
        if (w.shape != (len(models), len(names)) or not np.isfinite(w).all() or (w < 0).any()
                or not np.allclose(w.sum(axis=0), 1.)):
            raise ValueError("invalid per-class weights")
        combined = (p * w[:, None, :]).sum(axis=0)
        if (combined.sum(axis=1) <= 0).any():
            raise ValueError("class-weighted vote has zero total evidence")
        combined /= combined.sum(axis=1, keepdims=True)
    return CandidateScores(combined, np.ones_like(combined, dtype=bool), tuple(names))


def fit_acceptance(candidates, y, target_precision=.95, minimum_accepted=20, recall_floor=.5, recall_floors=None):
    """R2: try precision+recall, else explicit recall backoff; no null ban.

    Infeasible constraints remain visible in diagnostics. Relaxing precision to
    preserve recall is a comparison policy, not a claim of recovered accuracy.
    """
    if not 0 <= recall_floor <= 1:
        raise ValueError("recall_floor must be in [0,1]")
    overrides = dict(recall_floors or {})
    if (set(overrides) - (set(candidates.class_names) - {"background"})
            or any(not 0 <= v <= 1 for v in overrides.values())):
        raise ValueError("invalid per-label recall floors")
    y = np.asarray(y)
    common, _ = fit_thresholds(candidates, y, target_precision, minimum_accepted)
    common_feasible = common is not None
    common = 0. if common is None else float(common)
    top, score = candidates.top()
    grid = np.unique(np.r_[np.arange(0., 1., .005), .999, 1.])
    labels, diagnostics = {}, []
    for cid, label in enumerate(candidates.class_names):
        if label == "background":
            continue
        label_floor = float(overrides.get(label, recall_floor))
        support = int((y == cid).sum())
        possibilities = []
        for tau in grid:
            accept = (top == cid) & (score >= tau)
            n = int(accept.sum())
            tp = int((accept & (y == cid)).sum())
            precision, recall = (tp / n if n else 0.), (tp / support if support else 0.)
            possibilities.append((float(tau), n, tp, precision, recall))
        raw_recall = possibilities[0][4]
        good = [r for r in possibilities if r[1] >= minimum_accepted and r[3] >= target_precision and r[4] >= label_floor]
        if support < minimum_accepted:
            tau, reason = common, "low_support_shared_unverified"
        elif good:
            selected = max(good, key=lambda r: (r[2], -r[0]))
            tau, reason = selected[0], "precision_and_recall_fit"
        else:
            shared = next((r for r in possibilities if abs(r[0] - common) < 1e-9), None)
            if shared is not None and shared[4] >= label_floor:
                tau, reason = common, "shared_fallback_precision_unverified"
            else:
                recall_ok = [r for r in possibilities if r[4] >= label_floor and r[1] > 0]
                if recall_ok:
                    selected = max(recall_ok, key=lambda r: (r[3], r[2], r[0]))
                    tau, reason = selected[0], "recall_backoff_precision_relaxed"
                else:
                    tau, reason = 0., "upstream_recall_limit_no_extra_rejection"
        labels[label] = float(tau)
        accepted = (top == cid) & (score >= tau)
        n, tp = int(accepted.sum()), int((accepted & (y == cid)).sum())
        diagnostics.append(dict(label=label, threshold=float(tau), threshold_source=reason,
                                fit_support=support, fit_accepted=n,
                                fit_precision=tp / n if n else None,
                                fit_recall=tp / support if support else None,
                                raw_candidate_recall=raw_recall,
                                recall_floor=label_floor,
                                target_precision_met=bool(n >= minimum_accepted and tp / n >= target_precision),
                                recall_floor_met=bool(support > 0 and tp / support >= label_floor)))
    return dict(global_threshold=common, global_fit_feasible=common_feasible,
                class_thresholds=labels, diagnostics=diagnostics,
                target_precision=target_precision, recall_floor=recall_floor,
                label_recall_floors=overrides)


def decide_rule(candidates, rule, fitted):
    if rule not in RULES:
        raise ValueError(rule)
    return decide(candidates, {"R0": "S0", "R1": "S1", "R2": "S2"}[rule],
                  fitted["global_threshold"], fitted["class_thresholds"])


def cutoff_for(label_id, names, rule, fitted):
    return 0. if rule == "R0" else (fitted["global_threshold"] if rule == "R1"
                                    else fitted["class_thresholds"][names[label_id]])


def certify_partial(p, ready, weights, normalize_classes, names, rule, fitted):
    """Exact decision certificate for fixed nonnegative linear voting.

    Missing categorical probabilities lie in [0,1]. Certify only if every
    possible completion has the same top label and acceptance decision.
    This deterministic bound is our proposal, not Ringel/Guo's guarantee.
    """
    p, ready, w = np.asarray(p), np.asarray(ready, dtype=bool), np.asarray(weights)
    if p.shape != w.shape or ready.shape != (len(p),) or not ready.any():
        raise ValueError("invalid partial vote")
    if not np.isfinite(w).all() or (w < 0).any() or not np.allclose(w.sum(axis=0), 1.):
        raise ValueError("fixed column-normalized nonnegative weights required")
    known = p[ready]
    if (not np.isfinite(known).all() or (known < 0).any() or (known > 1).any()
            or not np.allclose(known.sum(axis=1), 1.)):
        raise ValueError("invalid ready model probabilities")
    lower_raw = (known * w[ready]).sum(axis=0)
    upper_raw = lower_raw + w[~ready].sum(axis=0)
    c = int(lower_raw.argmax())
    others = np.arange(len(names)) != c
    if not ready.all() and not lower_raw[c] > upper_raw[others].max() + 1e-12:
        return None
    if normalize_classes:
        lo = float(lower_raw[c] / max(lower_raw[c] + upper_raw[others].sum(), 1e-15))
        hi = float(upper_raw[c] / max(upper_raw[c] + lower_raw[others].sum(), 1e-15))
    else:
        lo, hi = float(lower_raw[c]), float(upper_raw[c])
    if names[c] == "background":
        return dict(top_id=c, accepted=False, status="background", score_lower=lo, score_upper=hi)
    tau = cutoff_for(c, names, rule, fitted)
    if lo >= tau + (0. if ready.all() else 1e-12):
        return dict(top_id=c, accepted=True, status="accepted", score_lower=lo, score_upper=hi)
    if hi < tau - (0. if ready.all() else 1e-12):
        return dict(top_id=c, accepted=False, status="unconfirmed", score_lower=lo, score_upper=hi)
    return None


@dataclass(frozen=True)
class AdaptiveHop:
    fast: float = .5
    slow: float = 1.
    confidence: float = .75
    margin: float = .2
    urgent: tuple[str, ...] = ("siren", "fire_alarm", "scream", "cry", "car_horn")

    def choose(self, scores, names, eligible=None):
        p = np.asarray(scores)
        if (not 0 < self.fast <= self.slow or p.shape != (len(names),)
                or not np.isfinite(p).all() or (p < 0).any() or (p > 1).any()
                or not 0 <= self.confidence <= 1 or not 0 <= self.margin <= 1):
            raise ValueError("invalid adaptive hop state")
        mask = np.ones(len(p), dtype=bool) if eligible is None else np.asarray(eligible, dtype=bool)
        if mask.shape != p.shape or not mask.any():
            raise ValueError("invalid adaptive eligibility")
        order = np.argsort(-np.where(mask, p, -np.inf), kind="stable")
        top = int(order[0])
        gap = float(p[top] - p[order[1]]) if mask.sum() > 1 else 1.
        difficult = p[top] < self.confidence or gap < self.margin
        return self.fast if difficult or names[top] in self.urgent else self.slow


class TemporalConfirmation:
    """T0 immediate, T1 two updates, T2 event-aware duration + hysteresis.

    Parameters are provisional experiment values. No new evidence independence
    is claimed for repeated overlapping windows. T2 is not an optimal policy.
    """
    def __init__(self, names, kind="T0", confirm_seconds=.5, release_floor=.2,
                 off_ratio=.8, strong_score=.9, strong_margin=.2, max_age=2.,
                 transient=("gun", "glass_breaking", "knock")):
        if kind not in ("T0", "T1", "T2"):
            raise ValueError(kind)
        if (not all(math.isfinite(x) for x in (confirm_seconds, release_floor, off_ratio, strong_score, strong_margin, max_age))
                or confirm_seconds < 0 or max_age < 0
                or not all(0 <= x <= 1 for x in (release_floor, off_ratio, strong_score, strong_margin))):
            raise ValueError("invalid temporal policy parameters")
        self.names, self.kind = tuple(names), kind
        self.confirm_seconds, self.release_floor, self.off_ratio = confirm_seconds, release_floor, off_ratio
        self.strong_score, self.strong_margin, self.max_age = strong_score, strong_margin, max_age
        self.transient = set(transient)
        self.active = self.pending = None
        self.since = None
        self.count = 0
        self.last_time = -math.inf

    def step(self, now, audio_end, scores, top, accepted, cutoff):
        if now < self.last_time or now < audio_end:
            raise ValueError("noncausal or unordered result")
        self.last_time = now
        p = np.asarray(scores)
        if p.shape != (len(self.names),) or not np.isfinite(p).all():
            raise ValueError("invalid temporal score vector")
        candidate = int(top) if accepted else None
        if self.kind == "T0":
            self.active = candidate
            return self.active, "immediate"
        if self.kind == "T2" and now - audio_end > self.max_age:
            self.active = self.pending = None
            self.count = 0
            return None, "stale_result"
        if candidate is None:
            self.pending, self.count = None, 0
        elif candidate == self.pending:
            self.count += 1
        else:
            self.pending, self.since, self.count = candidate, now, 1
        if self.kind == "T1":
            self.active = candidate if candidate is not None and self.count >= 2 else None
            return self.active, "two_updates"
        # T2 never retains a label against a new top-background result.
        if self.names[top] == "background":
            self.active = None
        elif self.active is not None:
            low = max(self.release_floor, self.off_ratio * cutoff(self.active))
            if p[self.active] < low:
                self.active = None
        if candidate is not None:
            other = np.delete(p, candidate)
            strong = p[candidate] >= self.strong_score and p[candidate] - other.max() >= self.strong_margin
            short = self.names[candidate] in self.transient
            confirmed = self.count >= 2 and now - self.since >= self.confirm_seconds - 1e-10
            if short or strong or confirmed:
                self.active = candidate
                return self.active, "transient" if short else ("strong" if strong else "duration_confirmed")
        return self.active, "hold_or_wait"


class PartialVoteCoordinator:
    """Single-consumer callback adapter for bounded same-window voting.

    Send callbacks through one event-loop thread. A certified output seals that
    window; later results are ignored, not cancelled. Newer emitted windows
    supersede older ones. Does not construct or run worker processes.
    """
    def __init__(self, config, fusion="F1", rule="R0", max_pending=32):
        if fusion not in ("F1", "F2") or rule not in RULES or max_pending < 1:
            raise ValueError("invalid partial coordinator config")
        self.names = config["class_names"]
        self.spec, self.fw = config["selection_spec"], config["framework"]
        self.models = self.fw["model_order"]
        if self.spec["score_type"] not in ("softmax_logits", "probabilities"):
            raise ValueError("unknown score type")
        if fusion not in self.fw["fusions"]:
            raise ValueError("fusion was not fitted")
        for m in self.models:
            if (len(self.spec["model_class_names"][m]) != len(self.names)
                    or set(self.spec["model_class_names"][m]) != set(self.names)):
                raise ValueError("partial voting requires complete common label heads")
        self.fusion, self.rule = fusion, rule
        self.fit = self.fw["acceptance"][fusion]
        self.weights = (np.full((len(self.models), len(self.names)), 1/len(self.models)) if fusion == "F1"
                        else np.array(self.fw["classwise_weights"], dtype=float))
        self.pending, self.last_emitted, self.stale_results = {}, -1, 0
        self.max_pending, self.dropped_windows = max_pending, []

    def submit(self, window_id, model_name, head_values, start_sample, end_sample):
        if model_name not in self.models or not 0 <= start_sample < end_sample:
            raise ValueError("unknown model or invalid input sample range")
        if window_id <= self.last_emitted:
            self.stale_results += 1
            return None
        values = np.asarray(head_values, dtype=float)
        head = self.spec["model_class_names"][model_name]
        if values.shape != (len(head),) or not np.isfinite(values).all():
            raise ValueError("invalid head output")
        p = softmax(values) if self.spec["score_type"] == "softmax_logits" else values
        if (p < 0).any() or (p > 1).any() or not np.isclose(p.sum(), 1.):
            raise ValueError("categorical head probabilities required")
        slot = self.pending.setdefault(int(window_id), dict(range=(start_sample, end_sample), outputs={}))
        if slot["range"] != (start_sample, end_sample) or model_name in slot["outputs"]:
            raise ValueError("mismatched input range or duplicate model callback")
        slot["outputs"][model_name] = values.copy()
        while len(self.pending) > self.max_pending:
            old = min(self.pending)
            self.dropped_windows.append(old)
            del self.pending[old]
        if window_id not in self.pending:
            return None
        ready = np.array([m in slot["outputs"] for m in self.models])
        if ready.all():
            outputs = {m: slot["outputs"][m][None] for m in self.models}
            full = fusion_candidates(outputs, self.names, self.spec["expert_classes"],
                                     self.spec["model_class_names"], self.fusion,
                                     self.spec["score_type"], self.weights)
            d = decide_rule(full, self.rule, self.fit)
            result = dict(top_id=int(d["top_id"][0]), accepted=bool(d["accepted"][0]), status=str(d["status"][0]),
                          score_lower=float(d["score"][0]), score_upper=float(d["score"][0]))
        else:
            visible = np.full((len(self.models), len(self.names)), np.nan)
            for i, m in enumerate(self.models):
                if ready[i]:
                    head = self.spec["model_class_names"][m]
                    v = slot["outputs"][m]
                    local = softmax(v) if self.spec["score_type"] == "softmax_logits" else v
                    visible[i] = local[[head.index(c) for c in self.names]]
            result = certify_partial(visible, ready, self.weights, self.fusion == "F2", self.names, self.rule, self.fit)
            if result is None:
                return None
        self.last_emitted = int(window_id)
        for old in [k for k in self.pending if k <= window_id]:
            if old != window_id:
                self.dropped_windows.append(old)
            del self.pending[old]
        result.update(window_id=int(window_id), start_sample=int(start_sample), end_sample=int(end_sample),
                      label=self.names[result["top_id"]], models_ready=int(ready.sum()),
                      display_label=self.names[result["top_id"]] if result["accepted"] else None,
                      model_jobs_cancelled=False)
        return result


class AdaptiveAudioBuffer:
    """Mono ring buffer with one full-result input request in flight (I3).

    Capture may continue during inference. `complete` consumes ONLY the full
    selected-window result, then requests the next eligible grid time. Output
    is latest-available if oversized chunks overrun the requested grid point.
    This adapter performs no microphone I/O, inference or thread management.
    """
    def __init__(self, names, sample_rate=16000, policy=None, base_hop=.5):
        self.names, self.sr = tuple(names), int(sample_rate)
        self.policy = policy or AdaptiveHop()
        self.base = round(base_hop * self.sr)
        self.window = 2 * self.sr
        if self.sr <= 0 or self.base <= 0 or self.base > self.window or self.window % self.base:
            raise ValueError("invalid buffer grid")
        for hop in (self.policy.fast, self.policy.slow):
            if not math.isfinite(hop) or hop <= 0 or round(hop * self.sr) < self.base or round(hop * self.sr) % self.base:
                raise ValueError("adaptive hops must be on base grid")
        self.total = 0
        self.buffer = np.empty(0, dtype=np.float32)
        self.next_end, self.inflight, self.next_id = self.window, None, 0
        self.skipped_grid_points = 0

    def _ready(self):
        if self.inflight is not None or self.total < self.next_end:
            return None
        available_end = self.total // self.base * self.base
        end = max(self.next_end, available_end)
        self.skipped_grid_points += max(0, (end - self.next_end) // self.base)
        buffer_start = self.total - len(self.buffer)
        local = end - self.window - buffer_start
        raw = self.buffer[local:local + self.window].copy()
        if len(raw) != self.window:
            raise RuntimeError("ring buffer lost requested input samples")
        raw.setflags(write=False)
        result = AudioWindow(self.next_id, end - self.window, end, self.sr, raw)
        self.inflight = (self.next_id, end)
        self.next_id += 1
        return result

    def push(self, mono_chunk):
        x = np.asarray(mono_chunk, dtype=np.float32)
        if x.ndim != 1 or not np.isfinite(x).all():
            raise ValueError("finite mono audio required")
        self.total += len(x)
        self.buffer = np.concatenate((self.buffer, x))[-(self.window + self.base):].copy()
        return self._ready()

    def complete(self, window_id, scores, eligible=None):
        if self.inflight is None or self.inflight[0] != window_id:
            raise ValueError("result does not match the full-result window in flight")
        _, previous_end = self.inflight
        hop = round(self.policy.choose(scores, self.names, eligible) * self.sr)
        self.next_end = int(math.ceil(max(previous_end + hop, self.total) / self.base)) * self.base
        self.inflight = None
        return self._ready()
