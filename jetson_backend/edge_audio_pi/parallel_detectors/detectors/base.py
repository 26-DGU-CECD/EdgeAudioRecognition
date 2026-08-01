"""
Common detector interface.

Every detector exposes the SAME minimal contract so `run_inference.py` can drive
any of them without special-casing:

    detector.target_classes            -> list[str]  (canonical class names, ordered)
    detector.predict_proba_batch(wavs) -> np.ndarray [N, len(target_classes)]

`wavs` is a float32 numpy array of shape [N, CLIP_SAMPLES] at 16 kHz mono,
amplitude in [-1, 1]. Each detector internally resamples/preprocesses as needed
and returns independent sigmoid probabilities in [0, 1] for its own classes.

Detectors that expose their pre-sigmoid outputs additionally provide:

    detector.provides_logits           -> bool
    detector.predict_logits_batch(w)   -> np.ndarray [N, len(target_classes)]
    detector.probs_from_logits(logits) -> np.ndarray [N, len(target_classes)]

`probs_from_logits` is THE single place where a logit becomes a probability, for
every detector and every caller (offline scoring, training val loops, the Pi
realtime engine, the calibration writer). Probability calibration is applied there
and nowhere else — see `parallel_detectors/calibrate.py`.
"""
from __future__ import annotations

import hashlib
import json
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np


class LogitsUnavailable(RuntimeError):
    """Detector has no recoverable pre-sigmoid output.

    Only YAMNet in zero-shot mode: TF-Hub YAMNet emits post-sigmoid scores, and the
    'mean' frame aggregation is not monotone-invertible.
    """


class CalibrationMismatch(RuntimeError):
    """A calibration was fitted for a different checkpoint than the one loaded."""


def _sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


class Calibration:
    """Per-class probability calibration: p = sigmoid((logit - bias) / temperature).

    Strictly increasing in `logit` for temperature > 0, so it leaves the per-class
    ranking — and therefore ROC-AUC, PR-AUC/AP and the best achievable F1 — exactly
    unchanged. It buys interpretable, numerically well-conditioned, cross-class
    comparable thresholds, not better detection. See `calibrate.py`.
    """

    def __init__(self, classes: list[str], temperature, bias, *,
                 source: str | None = None, checkpoint_sha256: str | None = None,
                 clamped: dict[str, bool] | None = None):
        self.classes = list(classes)
        self.temperature = np.asarray([float(temperature[c]) for c in self.classes],
                                      dtype=np.float64)
        self.bias = np.asarray([float(bias[c]) for c in self.classes], dtype=np.float64)
        if not np.all(self.temperature > 0):
            raise ValueError(f"temperature must be > 0, got {self.temperature}")
        self.source = source
        self.checkpoint_sha256 = checkpoint_sha256
        self.clamped = dict(clamped or {c: False for c in self.classes})

    def apply(self, logits: np.ndarray) -> np.ndarray:
        """[N, K] raw logits -> [N, K] calibrated logits, class order = self.classes."""
        z = np.asarray(logits, dtype=np.float64)
        if z.shape[-1] != len(self.classes):
            raise ValueError(f"expected {len(self.classes)} classes, got {z.shape[-1]}")
        return (z - self.bias) / self.temperature

    @property
    def id(self) -> str:
        """Short stable hash of (classes, T, b) — recorded in scores CSVs and
        thresholds.json so calibrated and uncalibrated numbers can never be mixed."""
        payload = json.dumps({
            "classes": self.classes,
            "temperature": [round(float(t), 10) for t in self.temperature],
            "bias": [round(float(b), 10) for b in self.bias],
        }, sort_keys=True)
        return hashlib.sha256(payload.encode()).hexdigest()[:8]

    def to_dict(self) -> dict:
        return {
            "classes": self.classes,
            "temperature": {c: float(t) for c, t in zip(self.classes, self.temperature)},
            "bias": {c: float(b) for c, b in zip(self.classes, self.bias)},
            "clamped": self.clamped,
            "checkpoint_sha256": self.checkpoint_sha256,
            "id": self.id,
        }

    @classmethod
    def load(cls, detector: str, classes: list[str], path=None, ckpt_path=None,
             required: bool = False) -> "Calibration | None":
        """Load this detector's calibration, or None if there isn't one.

        Refuses (does not warn) when the stored checkpoint hash disagrees with the
        checkpoint actually loaded: applying one model's calibration to another is
        silently wrong and otherwise undetectable.
        """
        # Relative, like every sibling detector module: the deploy bundle is imported
        # as `edge_audio_pi.parallel_detectors`, where an absolute import would fail.
        from .. import config

        path = Path(path) if path is not None else config.CALIBRATION_JSON
        if not path.exists():
            if required:
                raise FileNotFoundError(f"calibration file not found: {path}")
            return None
        blob = json.loads(path.read_text())
        entry = blob.get("detectors", {}).get(detector)
        if entry is None:
            if required:
                raise KeyError(f"{path} has no calibration for detector {detector!r}")
            return None
        if list(entry["classes"]) != list(classes):
            raise CalibrationMismatch(
                f"{path}: calibration for {detector} covers {entry['classes']}, but the "
                f"loaded model has {list(classes)}. Re-run calibrate.py.")
        stored_sha = entry.get("checkpoint_sha256")
        if stored_sha and ckpt_path is not None and Path(ckpt_path).exists():
            actual = _sha256(ckpt_path)
            if actual != stored_sha:
                raise CalibrationMismatch(
                    f"{path}: calibration for {detector} was fitted on a different "
                    f"checkpoint (stored sha256 {stored_sha[:12]}…, {ckpt_path} is "
                    f"{actual[:12]}…). Re-run calibrate.py against this checkpoint, or "
                    f"pass calibration=False.")
        obj = cls(entry["classes"], entry["temperature"], entry["bias"],
                  source=str(path), checkpoint_sha256=stored_sha,
                  clamped=entry.get("clamped"))
        bad = [c for c, f in obj.clamped.items() if f]
        banner = (f"[calib] {detector}: id={obj.id} from {path.name} "
                  f"(checkpoint sha {'ok' if stored_sha else 'unverified'})")
        print(banner)
        for c, t, b in zip(obj.classes, obj.temperature, obj.bias):
            print(f"         {c:14s} T={t:8.4f}  b={b:+9.4f}"
                  + ("   [CLAMPED - treat as uncalibrated]" if obj.clamped.get(c) else ""))
        if bad:
            print(f"[calib] WARNING: {detector} classes {bad} hit the temperature bound; "
                  f"their probabilities are NOT calibrated.")
        return obj


class BaseDetector(ABC):
    #: canonical class names this detector is responsible for, in output order
    target_classes: list[str]

    #: True when `predict_logits_batch` is implemented (pre-sigmoid outputs exist)
    provides_logits: bool = False

    #: per-class temperature/bias, applied inside `probs_from_logits`
    calibration: Calibration | None = None

    @abstractmethod
    def predict_proba_batch(self, waveforms_16k: np.ndarray) -> np.ndarray:
        """Return [N, len(target_classes)] sigmoid probabilities for a batch.

        Args:
            waveforms_16k: float32 array [N, CLIP_SAMPLES] @ 16 kHz mono, [-1, 1].
        """
        raise NotImplementedError

    def predict_logits_batch(self, waveforms_16k: np.ndarray) -> np.ndarray:
        """Return [N, len(target_classes)] RAW (uncalibrated) logits for a batch."""
        raise LogitsUnavailable(f"{type(self).__name__} exposes no logits")

    def probs_from_logits(self, logits: np.ndarray) -> np.ndarray:
        """Raw logits -> probabilities. The one place calibration is applied.

        float64 throughout: raw logits reach ±40 on the saturated detectors, and
        float32 sigmoid rounds those to exactly 1.0 / 0.0, destroying the margin.
        """
        z = np.asarray(logits, dtype=np.float64)
        if self.calibration is not None:
            z = self.calibration.apply(z)
        return (1.0 / (1.0 + np.exp(-z))).astype(np.float32)

    def _init_calibration(self, detector_name: str, calibration, ckpt_path=None) -> None:
        """Resolve the `calibration` constructor argument.

        False/None-with-no-file -> identity; True or a path -> load and verify;
        default (unset) -> auto-load `report/calibration.json` if it has an entry.
        """
        if calibration is False:
            self.calibration = None
            return
        if isinstance(calibration, Calibration):
            self.calibration = calibration
            return
        path = None if calibration in (None, True) else calibration
        self.calibration = Calibration.load(
            detector_name, self.target_classes, path=path, ckpt_path=ckpt_path,
            required=calibration is True or (path is not None),
        )
