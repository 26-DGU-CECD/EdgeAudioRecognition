from __future__ import annotations

import sys
import time
from datetime import datetime

import numpy as np

from angle_utils import angle_to_cardinal, corrected_angle
from audio_buffer import AudioBuffer
from audio_level_meter import AudioLevelMeter
from audio_preprocessor import AudioPreprocessor
from audio_queue import AudioQueue
from ble_inference_server import AppBleInferenceServer
from cli import format_optional_db_gate
from db_threshold_gate import DbThresholdGate
from doa_selector import DOASelector
from microphone_module import MicrophoneModule
from model_runner import SoundRecognitionModel
from packet_builder import AppSoundPacketBuilder, app_db_from_dbfs
from realtime_inference import (
    ANSI_GREEN,
    ANSI_RED,
    CHUNK_SAMPLES,
    CHUNK_SECONDS,
    MODEL_INPUT_SECONDS,
    MODEL_SAMPLE_RATE,
    REQUIRED_INPUT_CHANNELS,
    SAMPLE_RATE,
    colorize,
    db_gate_threshold,
    format_scores,
)


class AudioStreamController:
    """마이크 수집부터 버퍼, dB 판단, 전처리, 모델 추론, BLE publish까지 전체 흐름 제어."""

    def __init__(
        self,
        *,
        device_index: int,
        device_info: dict,
        stream_channels: int,
        channel_index: int,
        debug: bool,
        min_score: float,
        min_db: float,
        ble_server: AppBleInferenceServer,
        microphone: MicrophoneModule,
        audio_queue: AudioQueue,
        audio_buffer: AudioBuffer,
        level_meter: AudioLevelMeter,
        threshold_gate: DbThresholdGate,
        preprocessor: AudioPreprocessor,
        model: SoundRecognitionModel,
        doa_selector: DOASelector,
        packet_builder: AppSoundPacketBuilder,
        audio_doa_min_dbfs: float | None,
        usb_doa_status: str,
        audio_doa_status: str,
        doa_source: str,
        north_offset: float,
        db_offset: float,
        skip_inference_below_threshold: bool = True,
    ) -> None:
        self.device_index = device_index
        self.device_info = device_info
        self.stream_channels = stream_channels
        self.channel_index = channel_index
        self.debug = debug
        self.min_score = float(min_score)
        self.min_db = float(min_db)
        self.ble_server = ble_server
        self.microphone = microphone
        self.audio_queue = audio_queue
        self.audio_buffer = audio_buffer
        self.level_meter = level_meter
        self.threshold_gate = threshold_gate
        self.preprocessor = preprocessor
        self.model = model
        self.doa_selector = doa_selector
        self.packet_builder = packet_builder
        self.audio_doa_min_dbfs = audio_doa_min_dbfs
        self.usb_doa_status = usb_doa_status
        self.audio_doa_status = audio_doa_status
        self.doa_source = doa_source
        self.north_offset = float(north_offset)
        self.db_offset = float(db_offset)
        self.skip_inference_below_threshold = skip_inference_below_threshold
        self.is_running = False

    def _validate(self) -> None:
        if self.channel_index < 0 or self.channel_index >= self.stream_channels:
            raise RuntimeError(
                f"Selected channel index is out of range: "
                f"channel={self.channel_index}, available=0..{self.stream_channels - 1}"
            )

        if self.stream_channels < REQUIRED_INPUT_CHANNELS:
            print(
                f"Warning: selected input device reports only {self.stream_channels} input channels. "
                "Continuing with the available channel.",
                file=sys.stderr,
                flush=True,
            )

    def _audio_callback(self, indata, frames, time_info, status) -> None:  # noqa: ANN001
        if status:
            print(f"Audio input status: {status}", file=sys.stderr, flush=True)
        self.audio_queue.push(indata)

    def _print_start_info(self) -> None:
        print(
            f"Input device: [{self.device_index}] {self.device_info.get('name')} | "
            f"channels={self.stream_channels}, mic_sr={SAMPLE_RATE}, "
            f"model_sr={MODEL_SAMPLE_RATE}, chunk={CHUNK_SECONDS}s, "
            f"model_input={MODEL_INPUT_SECONDS}s, channel={self.channel_index}, "
            f"min_dbfs={db_gate_threshold(self.min_db):+.1f}, "
            f"enhance_threshold_dbfs={db_gate_threshold(self.preprocessor.enhance_threshold_db):+.1f}, "
            f"noise_reduction_db={self.preprocessor.noise_reduction_db:.1f}, "
            f"main_gain_db={self.preprocessor.main_gain_db:+.1f}, "
            f"min_score={self.min_score:.1%}, doa_source={self.doa_source}, "
            f"audio_doa_min_dbfs={format_optional_db_gate(self.audio_doa_min_dbfs)}, "
            f"usb_doa={self.usb_doa_status}, audio_doa={self.audio_doa_status}"
        )
        print("Ctrl+C to stop.")

    def start_stream(self) -> None:
        self._validate()
        self._print_start_info()
        self.is_running = True
        self.microphone.open(self._audio_callback)

    def stop_stream(self) -> None:
        self.is_running = False
        self.microphone.close()

    def process_loop(self) -> None:
        self.start_stream()
        try:
            while self.is_running:
                block = self.audio_queue.pop()
                self.audio_buffer.append(block)

                if not self.audio_buffer.is_chunk_ready():
                    continue

                self.audio_buffer.prepare_joined_blocks()
                while self.audio_buffer.has_next_chunk():
                    self._process_one_chunk(self.audio_buffer.get_chunk())
                self.audio_buffer.clear_processed()
        finally:
            self.stop_stream()

    def _process_one_chunk(self, chunk_multi: np.ndarray) -> None:
        chunk_started = time.perf_counter()
        chunk = chunk_multi[:, self.channel_index].astype(np.float32, copy=True)
        timestamp = datetime.now().strftime("%H:%M:%S")

        chunk_dbfs = self.level_meter.calculate_dbfs(chunk)
        if self.skip_inference_below_threshold and not self.threshold_gate.is_over_threshold(chunk_dbfs):
            print(
                colorize(
                    f"[{timestamp}] skip: {self.threshold_gate.low_signal_reason(chunk_dbfs)} | "
                    f"app_db={app_db_from_dbfs(chunk_dbfs, self.db_offset):.1f} dB",
                    ANSI_RED,
                ),
                flush=True,
            )
            return

        preprocess_result = self.preprocessor.preprocess(chunk)
        inference_chunk = preprocess_result.processed_audio

        try:
            infer_started = time.perf_counter()
            best_label, best_probability, scores = self.model.predict(inference_chunk, debug=self.debug)
            infer_sec = time.perf_counter() - infer_started
        except Exception as exc:
            print(
                f"[{timestamp}] inference error: {exc} | skipping chunk",
                file=sys.stderr,
                flush=True,
            )
            return

        status_reasons = []
        if not self.threshold_gate.is_over_threshold(chunk_dbfs):
            status_reasons.append(self.threshold_gate.low_signal_reason(chunk_dbfs))
        if best_probability < self.min_score:
            status_reasons.append(f"low_score {best_probability:.1%}<{self.min_score:.1%}")

        if status_reasons:
            status_text = "low(" + ", ".join(status_reasons) + ")"
            line_color = ANSI_RED
        else:
            status_text = "detected"
            line_color = ANSI_GREEN

        doa_reading = self.doa_selector.choose(chunk_multi)
        raw_angle = doa_reading.raw_angle
        angle = corrected_angle(raw_angle, self.north_offset) if raw_angle is not None else None
        direction = angle_to_cardinal(angle) if angle is not None else ""
        if angle is None:
            doa_text = f" | DOA=unavailable source={doa_reading.source} status={doa_reading.status}"
        else:
            doa_text = (
                f" | DOA={direction} {angle}deg raw={raw_angle} "
                f"source={doa_reading.source} status={doa_reading.status}"
            )

        line = (
            f"[{timestamp}] predict: {best_label} ({best_probability:.1%}) | "
            f"status={status_text} | "
            f"level={chunk_dbfs:+.1f} dBFS | "
            f"app_db={app_db_from_dbfs(chunk_dbfs, self.db_offset):.1f} dB | "
            f"enhanced={preprocess_result.enhanced_dbfs:+.1f} dBFS | "
            f"infer={infer_sec:.3f}s | "
            f"quiet_gain={preprocess_result.quiet_gain:.2f}x "
            f"loud_gain={preprocess_result.loud_gain:.2f}x"
            f"{' clipped' if preprocess_result.is_clipped else ''}{doa_text} | all: {format_scores(scores)}"
        )
        print(colorize(line, line_color), flush=True)

        total_sec = CHUNK_SECONDS + (time.perf_counter() - chunk_started)
        self.ble_server.publish(
            self.packet_builder.build(
                timestamp=timestamp,
                label=best_label,
                score=best_probability,
                scores=scores,
                infer_sec=infer_sec,
                total_sec=total_sec,
                chunk_dbfs=chunk_dbfs,
                status_text=status_text,
                raw_line=line,
                raw_angle=raw_angle,
                doa_status=doa_reading.status,
                doa_source=doa_reading.source,
            )
        )
