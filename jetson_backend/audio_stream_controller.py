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
    build_app_packet,
    build_idle_packet,
    summarize_decision,
)
from db_threshold_gate import DbThresholdGate
from decision import Decision, DecisionGate
from doa import DISABLED_READING
from parallel_inference import InferenceResult, ParallelInferenceEngine

if TYPE_CHECKING:
    from battery_monitor import BatteryMonitor
    from doa import DOAReader
    from imu import IMUReader
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
        battery_monitor: BatteryMonitor | None = None,
        doa_reader: DOAReader | None = None,
        imu_reader: IMUReader | None = None,
        imu_swing_sign: float = 1.0,
        imu_max_sync_age: float = 0.25,
        db_offset: float = 90.0,
        north_offset: float = 0.0,
        full_packet: bool = False,
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
        self.battery_monitor = battery_monitor
        self.doa_reader = doa_reader
        self.imu_reader = imu_reader
        self.imu_swing_sign = 1.0 if float(imu_swing_sign) >= 0 else -1.0
        self.imu_max_sync_age = max(0.0, float(imu_max_sync_age))
        self.db_offset = float(db_offset)
        self.north_offset = float(north_offset)
        self.full_packet = bool(full_packet)
        self.skip_low_db = bool(skip_low_db)
        self.debug = bool(debug)
        self.max_windows_per_cycle = max(1, int(max_windows_per_cycle))
        self.dropped_windows = 0

    def print_startup_info(self) -> None:
        if self.microphone is not None:
            print(
                f"입력 디바이스: [{self.microphone.device_index}] "
                f"{self.microphone.device_info.get('name')} | "
                f"channels={self.microphone.stream_channels}, "
                f"channel={self.microphone.channel_index}, "
                f"sr={self.microphone.sample_rate}"
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
            f"ble={self.publisher is not None} | "
            f"battery={self.battery_monitor.describe() if self.battery_monitor else 'off'} | "
            f"doa={self.doa_reader.describe() if self.doa_reader else 'off'} "
            f"north_offset={self.north_offset:.0f} | "
            f"imu={self.imu_reader.describe() if self.imu_reader else 'off'} "
            f"swing_sign={self.imu_swing_sign:+.0f} | "
            f"db_offset={self.db_offset:+.0f} "
            f"(app db>={self.threshold_gate.min_dbfs + self.db_offset:.0f})",
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

        if self.skip_low_db and not self.threshold_gate.is_over_threshold(level_dbfs):
            line = (
                f"[{timestamp}] skip: low_signal | "
                f"level={level_dbfs:+.1f} dBFS < "
                f"{self.threshold_gate.min_dbfs:+.1f} dBFS"
            )
            print(colorize(line, "\033[31m"), flush=True)
            payload = build_idle_packet(
                timestamp=timestamp,
                reason="low_signal",
                chunk_dbfs=level_dbfs,
                db_offset=self.db_offset,
            )
            self._publish(payload)
            return payload

        started = time.perf_counter()
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

        line = self._format_line(timestamp, level_dbfs, result, decision)
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

        total_sec = time.perf_counter() - started

        if decision.best_label is None:
            # 임계값을 넘은 클래스가 없다. status:"ok"로 보내면 앱의 알림 규칙
            # (minNotificationScore=0.30)이 근접 후보로 오알림을 낼 수 있으므로
            # 배터리만 실어 나르는 idle 패킷으로 보낸다.
            payload = build_idle_packet(
                timestamp=timestamp,
                reason="below_threshold",
                chunk_dbfs=level_dbfs,
                db_offset=self.db_offset,
            )
            self._publish(payload)
            return payload

        label, score = summarize_decision(decision)
        reading = (
            self.doa_reader.snapshot() if self.doa_reader is not None
            else DISABLED_READING
        )
        swing_deg, imu_status = self._swing_for(reading)
        payload = build_app_packet(
            timestamp=timestamp,
            label=label,
            score=score,
            infer_sec=result.total_latency_ms / 1000.0,
            total_sec=total_sec,
            chunk_dbfs=level_dbfs,
            db_offset=self.db_offset,
            raw_line=(
                # 앱은 raw를 저장만 하고 표시하지 않는다. 콘솔 줄을 그대로 보내면
                # 한글 때문에 100바이트 넘게 먹으므로 압축한 ASCII 요약을 보낸다.
                f"{label} {score:.3f}/{self.decision_gate.threshold_for(label):.3f} "
                f"{level_dbfs:+.1f}dBFS {result.total_latency_ms:.0f}ms"
            ),
            probabilities=result.probabilities,
            raw_angle=reading.raw_angle,
            north_offset=self.north_offset,
            doa_status=reading.status,
            swing_deg=swing_deg,
            imu_status=imu_status,
            full_packet=self.full_packet,
        )
        self._publish(payload)
        return payload

    def _swing_for(self, reading) -> tuple[float | None, str]:  # noqa: ANN001
        """De-swing correction for this DOA reading, sampled at *its* instant.

        Returns `(None, status)` rather than 0.0 when there is no usable sample,
        so the console line can distinguish "corrected by 0 degrees" from "not
        corrected". A mismatch is reported as `nosync` instead of silently
        applying the most recent swing, which would belong to a different
        moment and add exactly the oscillation we are removing.
        """
        if self.imu_reader is None or not self.imu_reader.ok:
            return None, (
                self.imu_reader.describe() if self.imu_reader is not None else "disabled"
            )
        if reading.raw_angle is None:
            return None, self.imu_reader.describe()

        raw_swing = self.imu_reader.swing_at(
            reading.timestamp,
            max_age=self.imu_max_sync_age,
        )
        if raw_swing is None:
            return None, "nosync"

        swing = self.imu_swing_sign * raw_swing
        state = self.imu_reader.snapshot()
        print(
            f"[IMU] raw_doa={reading.raw_angle} swing={raw_swing:+.1f} "
            f"applied={swing:+.1f} yaw={state.yaw:.1f} ref={state.ref:.1f} "
            f"mean0.5s={state.gyro_mean:+.1f} "
            f"state={'TURN' if state.turning else 'still'}",
            file=sys.stderr,
            flush=True,
        )
        return swing, self.imu_reader.describe()

    def _format_line(
        self,
        timestamp: str,
        level_dbfs: float,
        result: InferenceResult,
        decision: Decision,
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
        return (
            f"[{timestamp}] {body} | level={level_dbfs:+.1f} dBFS | "
            f"{result.total_latency_ms:.0f}ms"
        )

    def _publish(self, data: dict) -> None:
        if self.battery_monitor is not None:
            data.update(self.battery_monitor.snapshot())
        if self.publisher is not None:
            self.publisher.publish(data)
