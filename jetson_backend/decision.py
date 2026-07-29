"""Apply independent per-class thresholds to detector probabilities."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List


@dataclass(frozen=True)
class Decision:
    candidates: List[str]
    new_events: List[str]
    best_label: str | None
    best_probability: float
    best_margin: float
    nearest_label: str
    nearest_probability: float


def load_thresholds(path: Path | str) -> Dict[str, float]:
    threshold_path = Path(path)
    if not threshold_path.is_file():
        raise RuntimeError(f"threshold 파일이 없습니다: {threshold_path}")

    with threshold_path.open(encoding="utf-8") as handle:
        document = json.load(handle)
    values = document.get("thresholds")
    if not isinstance(values, dict) or not values:
        raise ValueError(f"thresholds 객체가 없습니다: {threshold_path}")
    try:
        return {str(name): float(value) for name, value in values.items()}
    except (TypeError, ValueError) as exc:
        raise ValueError(f"threshold 값이 숫자가 아닙니다: {threshold_path}") from exc


class DecisionGate:
    def __init__(
        self,
        thresholds: Dict[str, float],
        *,
        classes: List[str],
        debounce_seconds: float = 3.0,
    ) -> None:
        self.thresholds = {str(name): float(value) for name, value in thresholds.items()}
        self.classes = list(classes)
        self.debounce_seconds = float(debounce_seconds)
        self._last_fired_at: Dict[str, float] = {}

        missing = [name for name in self.classes if name not in self.thresholds]
        if missing:
            raise ValueError(
                f"threshold가 없는 클래스: {missing}. "
                "thresholds.json과 현재 클래스 구성을 확인하세요."
            )

    def threshold_for(self, class_name: str) -> float:
        return self.thresholds[class_name]

    def evaluate(self, probabilities: Dict[str, float], now: float) -> Decision:
        missing = [name for name in self.classes if name not in probabilities]
        if missing:
            raise ValueError(f"확률이 없는 클래스: {missing}")

        ranked = sorted(
            (
                (
                    name,
                    float(probabilities[name]),
                    float(probabilities[name]) - self.thresholds[name],
                )
                for name in self.classes
            ),
            key=lambda item: (-item[2], item[0]),
        )
        candidates = [name for name, _probability, margin in ranked if margin >= 0.0]

        new_events: List[str] = []
        for name in candidates:
            last = self._last_fired_at.get(name)
            if last is None or now - last >= self.debounce_seconds:
                new_events.append(name)
            self._last_fired_at[name] = now

        nearest_name, nearest_probability, nearest_margin = ranked[0]
        best_label = candidates[0] if candidates else None
        best_probability = probabilities[best_label] if best_label else 0.0
        best_margin = (
            float(probabilities[best_label]) - self.thresholds[best_label]
            if best_label
            else nearest_margin
        )
        return Decision(
            candidates=candidates,
            new_events=new_events,
            best_label=best_label,
            best_probability=float(best_probability),
            best_margin=float(best_margin),
            nearest_label=nearest_name,
            nearest_probability=float(nearest_probability),
        )
