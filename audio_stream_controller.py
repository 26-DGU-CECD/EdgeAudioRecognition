from __future__ import annotations

import time
from datetime import datetime

from realtime_inference import (
    ANSI_GREEN,
    ANSI_RED,
    CHUNK_SECONDS,
    CHUNK_SAMPLES,
    MODEL_INPUT_SECONDS,
    MODEL_SAMPLE_RATE,
    SAMPLE_RATE,
    colorize,
    format_scores,
)

from audio_buffer import AudioBuffer
from audio_level_meter import AudioLevelMeter
from audio_preprocessor import AudioPreprocessor
from audio_queue import AudioQueue
from ble_inference_server import AppBleInferenceServer
from cli import format_optional_db_gate
from db_threshold_gate import DbThresholdGate
from direction_utils import angle_to_cardinal, corrected_angle
from doa_selector import DOASelector
from microphone_module import MicrophoneModule
from model_inference import ModelInferenceEngine
from packet_builder import AppSoundPacketBuilder, app_db_from_dbfs


class AudioStreamController:
    def __init__(
        self,
        *,
        microphone: MicrophoneModule,
        audio_queue: AudioQueue,
        level_meter: AudioLevelMeter,
        threshold_gate: DbThresholdGate,
        preprocessor: AudioPreprocessor,
        model_engine: ModelInferenceEngine,
        doa_selector: DOASelector,
        ble_server: AppBleInferenceServer,
        packet_builder: AppSoundPacketBuilder,
        min_score: float,
        skip_inference_below_threshold: bool,
    ) -> None:
        self.microphone = microphone
        self.audio_queue = audio_queue
        self.buffer = AudioBuffer(CHUNK_SAMPLES)
        self.level_meter = level_meter
        self.threshold_gate = threshold_gate
        self.preprocessor = preprocessor
        self.model_engine = model_engine
        self.doa_selector = doa_selector
        self.ble_server = ble_server
        self.packet_builder = packet_builder
        self.min_score = float(min_score)
        self.skip_inference_below_threshold = bool(skip_inference_below_threshold)

    def print_startup_info(self) -> None:
        audio_estimator = self.doa_selector.audio_estimator
        usb_reader = self.doa_selector.usb_reader
        print(
            f"Input device: [{self.microphone.device_index}] {self.microphone.device_info.get('name')} | "
            f"channels={self.microphone.stream_channels}, mic_sr={SAMPLE_RATE}, "
            f"model_sr={MODEL_SAMPLE_RATE}, chunk={CHUNK_SECONDS}s, "
            f"model_input={MODEL_INPUT_SECONDS}s, channel={self.microphone.channel_index}, "
            f"min_dbfs={self.threshold_gate.min_dbfs:+.1f}, "
            f"enhance_threshold_dbfs={self.preprocessor.enhance_threshold_db:+.1f}, "
            f"noise_reduction_db={self.preprocessor.noise_reduction_db:.1f}, "
            f"main_gain_db={self.preprocessor.main_gain_db:+.1f}, "
            f"min_score={self.min_score:.1%}, doa_source={self.doa_selector.doa_source}, "
            f"audio_doa_min_dbfs={format_optional_db_gate(audio_estimator.min_dbfs)}, "
            f"usb_doa={usb_reader.status}, audio_doa={audio_estimator.status}"
        )
        print("Ctrl+C to stop.")

    def run(self) -> None:
        self.print_startup_info()
        with self.microphone.open():
            while True:
                block = self.audio_queue.pop()
                self.buffer.append(block)
                for chunk_multi in self.buffer.pop_ready_chunks():
                    self._process_chunk(chunk_multi)

    def _process_chunk(self, chunk_multi) -> None:  # noqa: ANN001
        chunk_started = time.perf_counter()
        chunk = self.microphone.select_model_channel(chunk_multi)
        timestamp = datetime.now().strftime("%H:%M:%S")

        chunk_dbfs = self.level_meter.calculate_dbfs(chunk)
        over_threshold = self.threshold_gate.is_over_threshold(chunk_dbfs)

        if self.skip_inference_below_threshold and not over_threshold:
            line = (
                f"[{timestamp}] skipped: low_signal | "
                f"level={chunk_dbfs:+.1f} dBFS < {self.threshold_gate.min_dbfs:+.1f} dBFS"
            )
            print(colorize(line, ANSI_RED), flush=True)
            return

        preprocess_result = self.preprocessor.preprocess(chunk)

        try:
            infer_started = time.perf_counter()
            best_label, best_probability, scores = self.model_engine.predict(preprocess_result.audio)
            infer_sec = time.perf_counter() - infer_started
        except Exception as exc:
            print(f"[{timestamp}] inference error: {exc} | skipping chunk", flush=True)
            return

        status_reasons = []
        if not over_threshold:
            status_reasons.append(self.threshold_gate.reason(chunk_dbfs))
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
        angle = corrected_angle(raw_angle, self.packet_builder.north_offset) if raw_angle is not None else None
        direction = angle_to_cardinal(angle) if angle is not None else ""
        if angle is None:
            doa_text = f" | DOA=unavailable source={doa_reading.source} status={doa_reading.status}"
        else:
            doa_text = f" | DOA={direction} {angle}deg raw={raw_angle} source={doa_reading.source} status={doa_reading.status}"

        line = (
            f"[{timestamp}] predict: {best_label} ({best_probability:.1%}) | "
            f"status={status_text} | "
            f"level={chunk_dbfs:+.1f} dBFS | "
            f"app_db={app_db_from_dbfs(chunk_dbfs, self.packet_builder.db_offset):.1f} dB | "
            f"enhanced={preprocess_result.enhanced_dbfs:+.1f} dBFS | "
            f"infer={infer_sec:.3f}s | "
            f"quiet_gain={preprocess_result.quiet_gain:.2f}x loud_gain={preprocess_result.loud_gain:.2f}x"
            f"{' clipped' if preprocess_result.clipped else ''}{doa_text} | all: {format_scores(scores)}"
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
