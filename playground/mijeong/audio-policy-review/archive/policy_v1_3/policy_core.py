"""Small policy pilot. NumPy only; original checkpoints and artifacts stay read-only."""
from __future__ import annotations
from dataclasses import dataclass
import math
from typing import Mapping, Sequence
import numpy as np

MODEL_NAMES = ("efficientat", "yamnet", "mobilenetv4")
INPUT_POLICIES = {"I0": 2.0, "I1": 1.0, "I2": 0.5}


@dataclass(frozen=True)
class AudioWindow:
    window_id: int
    start_sample: int
    end_sample: int
    sample_rate: int
    waveform: np.ndarray

    @property
    def end_seconds(self) -> float:
        return self.end_sample / self.sample_rate


class SlidingWindowScheduler:
    """Raw mono windows; each window uses only samples already received.

    Caller retains the current model preprocessing and submits each window
    to the three resident specialist workers. Large chunks are supported.
    """
    def __init__(self, sample_rate=16000, window_seconds=2.0, hop_seconds=2.0):
        self.sample_rate = int(sample_rate)
        if self.sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        if not all(math.isfinite(x) and x > 0 for x in (window_seconds, hop_seconds)):
            raise ValueError("window_seconds and hop_seconds must be positive and finite")
        self.window_samples = int(round(self.sample_rate * window_seconds))
        self.hop_samples = int(round(self.sample_rate * hop_seconds))
        if not 0 < self.hop_samples <= self.window_samples:
            raise ValueError("hop must be at most the window length")
        self.total_samples = 0
        self._buffer_start = 0
        self._buffer = np.empty(0, dtype=np.float32)
        self._next_end = self.window_samples
        self._next_id = 0

    def push(self, chunk: np.ndarray) -> list[AudioWindow]:
        chunk = np.asarray(chunk, dtype=np.float32)
        if chunk.ndim != 1 or not np.all(np.isfinite(chunk)):
            raise ValueError("chunk must be finite mono samples")
        if chunk.size == 0:
            return []
        self._buffer = np.concatenate((self._buffer, chunk))
        self.total_samples += chunk.size
        ready = []
        while self._next_end <= self.total_samples:
            start = self._next_end - self.window_samples
            local = start - self._buffer_start
            waveform = self._buffer[local:local + self.window_samples].copy()
            if len(waveform) != self.window_samples:
                raise RuntimeError("input samples missing")
            waveform.setflags(write=False)
            ready.append(AudioWindow(
                self._next_id, start, self._next_end,
                self.sample_rate, waveform,
            ))
            self._next_id += 1
            self._next_end += self.hop_samples
        keep_start = max(self._buffer_start, self._next_end - self.window_samples)
        self._buffer = self._buffer[keep_start - self._buffer_start:].copy()
        self._buffer_start = keep_start
        return ready


def softmax(logits: np.ndarray) -> np.ndarray:
    x = np.asarray(logits, dtype=np.float64)
    if not np.all(np.isfinite(x)):
        raise ValueError("logits must be finite")
    x = x - x.max(axis=-1, keepdims=True)
    exp = np.exp(x)
    return exp / exp.sum(axis=-1, keepdims=True)


def fuse_logits(logits: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """logits [models, ..., classes], weights [models, classes].

    Reproduces ensemble_specialists.py including final class normalization.
    """
    z = np.asarray(logits, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    if z.ndim < 2 or w.shape != (z.shape[0], z.shape[-1]):
        raise ValueError("model/class dimensions do not match")
    if not np.all(np.isfinite(w)) or np.any(w < 0):
        raise ValueError("weights must be finite and nonnegative")
    if not np.allclose(w.sum(axis=0), 1.0, atol=1e-10):
        raise ValueError("weights must sum to one for each class")
    shape = (w.shape[0],) + (1,) * (z.ndim - 2) + (w.shape[-1],)
    combined = (softmax(z) * w.reshape(shape)).sum(axis=0)
    return combined / combined.sum(axis=-1, keepdims=True)


class OutputPolicy:
    """Always select global top-1 first; never replace it with a lower class."""
    def __init__(self, class_names: Sequence[str], kind: str,
                 global_threshold=0.6,
                 class_thresholds: Mapping[str, float | None] | None = None):
        self.class_names = tuple(class_names)
        if len(set(self.class_names)) != len(self.class_names):
            raise ValueError("class names must be unique")
        self.background_id = self.class_names.index("background")
        if kind not in ("O0", "O1", "O2"):
            raise ValueError(kind)
        if not 0 <= global_threshold <= 1:
            raise ValueError("threshold must be in [0,1]")
        self.kind = kind
        self.global_threshold = float(global_threshold)
        self.class_thresholds = dict(class_thresholds or {})
        if kind == "O2":
            expected = set(self.class_names) - {"background"}
            if set(self.class_thresholds) != expected:
                raise ValueError("O2 needs every foreground class")
            for value in self.class_thresholds.values():
                if value is not None and not 0 <= value <= 1:
                    raise ValueError("invalid class threshold")

    def decide(self, probabilities: np.ndarray) -> dict[str, np.ndarray]:
        p = np.asarray(probabilities, dtype=np.float64)
        if p.ndim != 2 or p.shape[1] != len(self.class_names):
            raise ValueError("expected [windows, classes]")
        if (not np.all(np.isfinite(p)) or np.any(p < 0)
                or not np.allclose(p.sum(axis=1), 1.0, atol=1e-8)):
            raise ValueError("invalid probability vectors")
        top = p.argmax(axis=1)
        score = p[np.arange(len(p)), top]
        cutoff = np.zeros(len(p))
        if self.kind == "O1":
            cutoff[:] = self.global_threshold
        elif self.kind == "O2":
            cutoff = np.array([
                math.inf if self.class_thresholds.get(self.class_names[i]) is None
                else self.class_thresholds[self.class_names[i]]
                for i in top
            ], dtype=np.float64)
        accepted = (top != self.background_id) & (score >= cutoff)
        # Background (model result) and unconfirmed (threshold rejection) differ.
        status = np.where(
            top == self.background_id, "background",
            np.where(accepted, "accepted", "unconfirmed"),
        )
        return {"top_id": top, "score": score, "accepted": accepted, "status": status}


def fit_class_thresholds(probabilities, y_true, class_names,
                         target_precision=0.95, minimum_accepted=20):
    """Exploratory class-wise fit. No formal risk-control claim.

    The fit uses only the provided calibration subset. None means the
    pilot found no qualifying threshold; it does not mean the model is broken.
    """
    p = np.asarray(probabilities, dtype=np.float64)
    y = np.asarray(y_true, dtype=np.int64)
    if p.ndim != 2 or p.shape != (len(y), len(class_names)):
        raise ValueError("shape mismatch")
    if not 0 < target_precision <= 1 or minimum_accepted < 1:
        raise ValueError("invalid fitting requirements")
    if np.any(y < 0) or np.any(y >= len(class_names)):
        raise ValueError("unknown target class")
    # Reuse policy validation of finite, normalized probabilities.
    OutputPolicy(class_names, "O0").decide(p)
    pred = p.argmax(axis=1)
    grid = np.unique(np.concatenate((np.arange(0, 1, 0.025), [0.99, 0.995])))
    thresholds, rows = {}, []
    for cid, label in enumerate(class_names):
        if label == "background":
            continue
        best = None
        for tau in grid:
            accepted = (pred == cid) & (p[:, cid] >= tau)
            n = int(accepted.sum())
            tp = int((accepted & (y == cid)).sum())
            precision = tp / n if n else 0.0
            if n >= minimum_accepted and precision >= target_precision:
                candidate = (tp, -float(tau), float(tau), n, precision)
                if best is None or candidate > best:
                    best = candidate
        thresholds[label] = None if best is None else best[2]
        rows.append({
            "label": label, "threshold": thresholds[label],
            "fit_true_clips": int((y == cid).sum()),
            "fit_accepted": 0 if best is None else best[3],
            "fit_accepted_precision": None if best is None else best[4],
        })
    return thresholds, rows


class WindowFusionCoordinator:
    """Single-consumer callback for asynchronous specialist results.

    Results must share a window ID and sample range. This pilot waits for
    all three models, exactly preserving the offline weighted fusion.
    Feed this object from one event-loop/result-consumer thread.
    """
    def __init__(self, weights, class_names, output_policy, max_pending=32):
        self.weights = np.asarray(weights, dtype=np.float64).copy()
        self.class_names = tuple(class_names)
        self.policy = output_policy
        if self.weights.shape != (len(MODEL_NAMES), len(self.class_names)):
            raise ValueError("invalid weight shape")
        if max_pending < 1:
            raise ValueError("max_pending must be positive")
        self.max_pending = int(max_pending)
        self.pending = {}
        self.last_emitted_id = -1
        self.dropped_window_ids = []
        self.stale_results = 0

    def submit(self, window_id, model_name, logits, start_sample, end_sample):
        if model_name not in MODEL_NAMES:
            raise ValueError("unknown model")
        window_id = int(window_id)
        if window_id <= self.last_emitted_id:
            self.stale_results += 1
            return None
        z = np.asarray(logits, dtype=np.float64)
        if z.shape != (len(self.class_names),) or not np.all(np.isfinite(z)):
            raise ValueError("invalid logits")
        if not 0 <= start_sample < end_sample:
            raise ValueError("invalid sample range")
        if window_id not in self.pending:
            self.pending[window_id] = {
                "range": (int(start_sample), int(end_sample)), "scores": {},
            }
        item = self.pending[window_id]
        if item["range"] != (start_sample, end_sample):
            raise ValueError("different input ranges in the same window")
        if model_name in item["scores"]:
            raise ValueError("duplicate model result")
        item["scores"][model_name] = z.copy()
        while len(self.pending) > self.max_pending:
            oldest = min(self.pending)
            self.dropped_window_ids.append(oldest)
            del self.pending[oldest]
        if window_id not in self.pending or len(item["scores"]) < len(MODEL_NAMES):
            return None
        probability = fuse_logits(
            np.stack([item["scores"][m] for m in MODEL_NAMES]), self.weights,
        )
        decision = self.policy.decide(probability[None, :])
        self.last_emitted_id = window_id
        for old in sorted(k for k in self.pending if k <= window_id):
            if old != window_id:
                self.dropped_window_ids.append(old)
            del self.pending[old]
        return {
            "window_id": window_id,
            "start_sample": int(start_sample), "end_sample": int(end_sample),
            "probabilities": probability,
            "top_id": int(decision["top_id"][0]),
            "score": float(decision["score"][0]),
            "accepted": bool(decision["accepted"][0]),
            "status": str(decision["status"][0]),
        }

