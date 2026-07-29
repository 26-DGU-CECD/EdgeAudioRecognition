"""Run independent specialized detectors over the same audio window."""
from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Dict

import numpy as np

import runtime_config


@dataclass(frozen=True)
class InferenceResult:
    probabilities: Dict[str, float]
    detector_latency_ms: Dict[str, float] = field(default_factory=dict)
    total_latency_ms: float = 0.0


class ParallelInferenceEngine:
    def __init__(
        self,
        detectors: Dict[str, object],
        *,
        concurrent: bool = False,
        window_samples: int = runtime_config.WINDOW_SAMPLES,
    ) -> None:
        if not detectors:
            raise ValueError("최소 한 개의 detector가 필요합니다.")
        self.detectors = dict(detectors)
        self.window_samples = int(window_samples)
        self.concurrent = bool(concurrent) and len(detectors) > 1
        self._pool = (
            ThreadPoolExecutor(max_workers=len(detectors))
            if self.concurrent
            else None
        )

        classes: list[str] = []
        for detector in self.detectors.values():
            for class_name in detector.target_classes:
                if class_name in classes:
                    raise ValueError(f"검출기 간 중복 클래스: {class_name}")
                classes.append(class_name)
        self.classes = classes

    def _run_one(
        self,
        name: str,
        window: np.ndarray,
    ) -> tuple[Dict[str, float], float]:
        detector = self.detectors[name]
        started = time.perf_counter()
        output = detector.predict_proba_batch(window[None, :])
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        if output.shape != (1, len(detector.target_classes)):
            raise ValueError(
                f"{name} 출력 shape 오류: {output.shape}, "
                f"expected (1, {len(detector.target_classes)})"
            )
        scores = {
            class_name: float(output[0, index])
            for index, class_name in enumerate(detector.target_classes)
        }
        return scores, elapsed_ms

    def predict(self, window_16k: np.ndarray) -> InferenceResult:
        window = np.ascontiguousarray(
            np.asarray(window_16k, dtype=np.float32).reshape(-1)
        )
        if window.shape[0] != self.window_samples:
            raise ValueError(
                f"window must be exactly {self.window_samples} samples, "
                f"got {window.shape[0]}"
            )

        probabilities: Dict[str, float] = {}
        latency: Dict[str, float] = {}
        started = time.perf_counter()

        if self._pool is not None:
            futures = {
                name: self._pool.submit(self._run_one, name, window)
                for name in self.detectors
            }
            for name, future in futures.items():
                scores, elapsed = future.result()
                probabilities.update(scores)
                latency[name] = elapsed
        else:
            for name in self.detectors:
                scores, elapsed = self._run_one(name, window)
                probabilities.update(scores)
                latency[name] = elapsed

        return InferenceResult(
            probabilities=probabilities,
            detector_latency_ms=latency,
            total_latency_ms=(time.perf_counter() - started) * 1000.0,
        )

    def close(self) -> None:
        if self._pool is not None:
            self._pool.shutdown(wait=True)
            self._pool = None
