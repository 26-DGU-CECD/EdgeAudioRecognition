from __future__ import annotations

from typing import Dict

from decision import Decision
from motion_state import UNKNOWN_SNAPSHOT, MotionSnapshot

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
    motion: MotionSnapshot = UNKNOWN_SNAPSHOT,
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
        "motion": motion.as_dict(),
    }


def build_skip_result(
    *,
    timestamp: str,
    chunk_dbfs: float,
    threshold_dbfs: float,
    raw_line: str,
    motion: MotionSnapshot = UNKNOWN_SNAPSHOT,
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
        "motion": motion.as_dict(),
    }


def build_motion_skip_result(
    *,
    timestamp: str,
    chunk_dbfs: float,
    raw_line: str,
    motion: MotionSnapshot,
) -> dict:
    """움직임 때문에 추론을 건너뛴 윈도우.

    label은 움직임 상태를 그대로 쓴다. 앱에서 shock/free_fall은 소리와 별개로
    알릴 수 있고, motion은 그냥 무시하면 된다.
    """
    return {
        "source": SOURCE,
        "time": timestamp,
        "label": motion.state,
        "score": 0.0,
        "status": "motion_skipped",
        "status_text": f"움직임감지 {motion.state}",
        "level_dbfs": round(float(chunk_dbfs), 2),
        "enhanced_dbfs": None,
        "quiet_gain": None,
        "loud_gain": None,
        "clipped": False,
        "scores": {},
        "raw": raw_line,
        "candidates": [],
        "repeat": False,
        "motion": motion.as_dict(),
    }
