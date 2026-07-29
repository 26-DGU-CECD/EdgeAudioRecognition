from __future__ import annotations

from typing import Dict

from decision import Decision

LEGACY_KEYS = {
    "source",
    "time",
    "label",
    "score",
    "status",
    "status_text",
    "level_dbfs",
    "enhanced_dbfs",
    "quiet_gain",
    "loud_gain",
    "clipped",
    "scores",
    "raw",
}

SOURCE = "live_inference_refactored_ble_independent"


def build_ble_result(
    *,
    timestamp: str,
    probabilities: Dict[str, float],
    decision: Decision,
    chunk_dbfs: float,
    raw_line: str,
) -> dict:
    detected = bool(decision.candidates)
    return {
        "source": SOURCE,
        "time": timestamp,
        "label": decision.best_label or "none",
        "score": round(float(decision.best_probability), 6),
        "status": "detected" if detected else "low_score",
        "status_text": "감지" if detected else "점수낮음",
        "level_dbfs": round(float(chunk_dbfs), 2),
        "enhanced_dbfs": None,
        "quiet_gain": None,
        "loud_gain": None,
        "clipped": False,
        "scores": {
            label: round(float(score), 6)
            for label, score in probabilities.items()
        },
        "raw": raw_line,
        "candidates": list(decision.candidates),
        "repeat": detected and not decision.new_events,
    }


def build_skip_result(
    *,
    timestamp: str,
    chunk_dbfs: float,
    threshold_dbfs: float,
    raw_line: str,
) -> dict:
    return {
        "source": SOURCE,
        "time": timestamp,
        "label": "low_signal",
        "score": 0.0,
        "status": "low_signal",
        "status_text": f"소리작음 {chunk_dbfs:+.1f}<{threshold_dbfs:+.1f}dBFS",
        "level_dbfs": round(float(chunk_dbfs), 2),
        "enhanced_dbfs": None,
        "quiet_gain": None,
        "loud_gain": None,
        "clipped": False,
        "scores": {},
        "raw": raw_line,
        "candidates": [],
        "repeat": False,
    }
