"""Realtime and WAV-file orchestration for the specialized detectors."""
from __future__ import annotations

import sys
import time
from datetime import datetime
from typing import TYPE_CHECKING, Protocol

import numpy as np

from audio_buffer import SlidingWindowBuffer
from audio_level_meter import AudioLevelMeter
from audio_math import colorize
from audio_queue import AudioQueue
from ble_result_builder import (
    build_ble_result,
    build_motion_skip_result,
    build_skip_result,
)
from db_threshold_gate import DbThresholdGate
from decision import Decision, DecisionGate
from motion_state import UNKNOWN_SNAPSHOT, MotionMonitor, MotionSnapshot
from parallel_inference import InferenceResult, ParallelInferenceEngine

if TYPE_CHECKING:
    from microphone_module import MicrophoneModule


class InferencePublisher(Protocol):
    def publish(self, data: dict) -> None: ...


class AudioStreamController:
    def __init__(
        self,
        *,
        audio_queue: AudioQueue,
        window_buffer: SlidingWindowBuffer,
        level_meter: AudioLevelMeter,
        threshold_gate: DbThresholdGate,
        inference_engine: ParallelInferenceEngine,
        decision_gate: DecisionGate,
        microphone: MicrophoneModule | None = None,
        publisher: InferencePublisher | None = None,
        motion_monitor: MotionMonitor | None = None,
        skip_low_db: bool = True,
        debug: bool = False,
        max_windows_per_cycle: int = 2,
    ) -> None:
        self.audio_queue = audio_queue
        self.window_buffer = window_buffer
        self.level_meter = level_meter
        self.threshold_gate = threshold_gate
        self.inference_engine = inference_engine
        self.decision_gate = decision_gate
        self.microphone = microphone
        self.publisher = publisher
        self.motion_monitor = motion_monitor
        self.skip_low_db = bool(skip_low_db)
        self.debug = bool(debug)
        self.max_windows_per_cycle = max(1, int(max_windows_per_cycle))
        self.dropped_windows = 0
        self.motion_suppressed_windows = 0

    def print_startup_info(self) -> None:
        if self.microphone is not None:
            print(
                f"입력 디바이스: [{self.microphone.device_index}] "
                f"{self.microphone.device_info.get('name')} | "
                f"channels={self.microphone.stream_channels}, "
                f"channel={self.microphone.channel_index}, "
                f"sr={self.microphone.sample_rate}"
            )
        if self.motion_monitor is not None and self.motion_monitor.enabled:
            print(
                f"IMU: {self.motion_monitor.imu.description} | "
                f"suppress_on_motion={self.motion_monitor.suppress_on_motion}"
            )
        hop_seconds = self.window_buffer.hop_samples / 16000
        print(
            f"window={self.window_buffer.window_samples / 16000:.1f}s "
            f"hop={hop_seconds:.2f}s | "
            f"detectors={list(self.inference_engine.detectors)} | "
            f"classes={len(self.inference_engine.classes)} | "
            f"min_dbfs={self.threshold_gate.min_dbfs:+.1f} "
            f"skip_low_db={self.skip_low_db} | "
            f"debounce={self.decision_gate.debounce_seconds:.1f}s | "
            f"ble={self.publisher is not None}",
            flush=True,
        )

    def run(self) -> None:
        if self.microphone is None:
            raise RuntimeError("마이크 없이 run()을 호출했습니다. run_file()을 사용하세요.")
        self.print_startup_info()
        print("Ctrl+C로 종료합니다.", flush=True)
        with self.microphone:
            while True:
                block = self.audio_queue.pop(timeout=1.0)
                if block is None:
                    continue
                dropped = self.audio_queue.take_dropped()
                if dropped:
                    print(
                        f"경고: 입력 큐 포화로 오디오 블록 {dropped}개 폐기",
                        file=sys.stderr,
                        flush=True,
                    )
                self.window_buffer.append(self.microphone.extract_mono(block))
                for window in self._take_windows():
                    self.process_window(window)

    def _take_windows(self) -> list[np.ndarray]:
        windows = self.window_buffer.pop_windows()
        if len(windows) > self.max_windows_per_cycle:
            self.dropped_windows += len(windows) - self.max_windows_per_cycle
            windows = windows[-self.max_windows_per_cycle :]
        return windows

    def run_file(self, waveform_16k: np.ndarray) -> int:
        self.print_startup_info()
        self.window_buffer.append(np.asarray(waveform_16k, dtype=np.float32))
        windows = self.window_buffer.pop_windows()
        for window in windows:
            self.process_window(window)
        if not windows:
            print("경고: 입력이 2초보다 짧아 처리된 윈도우가 없습니다.", file=sys.stderr)
        return len(windows)

    def process_window(self, window: np.ndarray) -> dict | None:
        timestamp = datetime.now().strftime("%H:%M:%S")
        level_dbfs = self.level_meter.calculate_dbfs(window)
        motion = self._motion_snapshot()

        if self.skip_low_db and not self.threshold_gate.is_over_threshold(level_dbfs):
            line = (
                f"[{timestamp}] skip: low_signal | "
                f"level={level_dbfs:+.1f} dBFS < "
                f"{self.threshold_gate.min_dbfs:+.1f} dBFS"
            )
            print(colorize(line, "\033[31m"), flush=True)
            payload = build_skip_result(
                timestamp=timestamp,
                chunk_dbfs=level_dbfs,
                threshold_dbfs=self.threshold_gate.min_dbfs,
                raw_line=line,
                motion=motion,
            )
            self._publish(payload)
            return payload

        # 기기를 만지는 중이면 마찰음이 그대로 들어오므로 추론 자체를 건너뛴다.
        # 판정 게이트를 거치지 않아 debounce 상태도 오염되지 않는다.
        if self.motion_monitor is not None and self.motion_monitor.should_suppress(motion):
            self.motion_suppressed_windows += 1
            line = (
                f"[{timestamp}] skip: {motion.state} | "
                f"level={level_dbfs:+.1f} dBFS | {motion.summary()}"
            )
            print(colorize(line, "\033[33m"), flush=True)
            payload = build_motion_skip_result(
                timestamp=timestamp,
                chunk_dbfs=level_dbfs,
                raw_line=line,
                motion=motion,
            )
            self._publish(payload)
            return payload

        try:
            result = self.inference_engine.predict(window)
            decision = self.decision_gate.evaluate(
                result.probabilities,
                time.monotonic(),
            )
        except Exception as exc:
            print(
                f"[{timestamp}] 추론 오류: {exc} | 해당 윈도우 skip",
                file=sys.stderr,
                flush=True,
            )
            return None

        line = self._format_line(timestamp, level_dbfs, result, decision, motion)
        color = "\033[32m" if decision.new_events else "\033[31m"
        print(colorize(line, color), flush=True)

        if self.debug:
            detail = ", ".join(
                f"{class_name}={probability:.3f}/"
                f"thr{self.decision_gate.threshold_for(class_name):.3f}"
                for class_name, probability in result.probabilities.items()
            )
            latency = ", ".join(
                f"{name}={elapsed:.0f}ms"
                for name, elapsed in result.detector_latency_ms.items()
            )
            print(f"DEBUG {detail} | {latency}", file=sys.stderr, flush=True)

        payload = build_ble_result(
            timestamp=timestamp,
            probabilities=result.probabilities,
            decision=decision,
            chunk_dbfs=level_dbfs,
            raw_line=line,
            motion=motion,
        )
        self._publish(payload)
        return payload

    def _motion_snapshot(self) -> MotionSnapshot:
        if self.motion_monitor is None:
            return UNKNOWN_SNAPSHOT
        return self.motion_monitor.snapshot()

    def _format_line(
        self,
        timestamp: str,
        level_dbfs: float,
        result: InferenceResult,
        decision: Decision,
        motion: MotionSnapshot,
    ) -> str:
        if decision.candidates:
            candidates = ", ".join(
                f"{class_name}={result.probabilities[class_name]:.3f}"
                f"(thr {self.decision_gate.threshold_for(class_name):.3f})"
                for class_name in decision.candidates
            )
            state = "감지" if decision.new_events else "감지(반복)"
            body = f"{state}: {decision.best_label} | 후보: {candidates}"
        else:
            body = (
                f"무발화 | 근접: {decision.nearest_label}="
                f"{decision.nearest_probability:.3f} "
                f"(thr {self.decision_gate.threshold_for(decision.nearest_label):.3f})"
            )
        motion_part = f" | {motion.summary()}" if motion.is_known else ""
        return (
            f"[{timestamp}] {body} | level={level_dbfs:+.1f} dBFS | "
            f"{result.total_latency_ms:.0f}ms{motion_part}"
        )

    def _publish(self, data: dict) -> None:
        if self.publisher is not None:
            self.publisher.publish(data)
