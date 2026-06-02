from __future__ import annotations

import sys
from datetime import datetime

from audio_buffer import AudioBuffer
from audio_level_meter import AudioLevelMeter
from audio_math import colorize, format_scores
from audio_preprocessor import AudioPreprocessor
from audio_queue import AudioQueue
from constants import (
    ANSI_GREEN,
    ANSI_RED,
    CHUNK_SECONDS,
    MODEL_INPUT_SECONDS,
    MODEL_SAMPLE_RATE,
    SAMPLE_RATE,
)
from db_threshold_gate import DbThresholdGate
from microphone_module import MicrophoneModule
from model_inference import ModelInferenceEngine


class AudioStreamController:
    def __init__(
        self,
        *,
        microphone: MicrophoneModule,
        audio_queue: AudioQueue,
        audio_buffer: AudioBuffer,
        level_meter: AudioLevelMeter,
        threshold_gate: DbThresholdGate,
        preprocessor: AudioPreprocessor,
        inference_engine: ModelInferenceEngine,
        min_score: float,
        skip_low_db: bool,
    ) -> None:
        self.microphone = microphone
        self.audio_queue = audio_queue
        self.audio_buffer = audio_buffer
        self.level_meter = level_meter
        self.threshold_gate = threshold_gate
        self.preprocessor = preprocessor
        self.inference_engine = inference_engine
        self.min_score = float(min_score)
        self.skip_low_db = bool(skip_low_db)

    def print_startup_info(self) -> None:
        print(
            f"입력 디바이스: [{self.microphone.device_index}] {self.microphone.device_info.get('name')} | "
            f"channels={self.microphone.stream_channels}, mic_sr={SAMPLE_RATE}, "
            f"model_sr={MODEL_SAMPLE_RATE}, chunk={CHUNK_SECONDS}s, "
            f"model_input={MODEL_INPUT_SECONDS}s, channel={self.microphone.channel_index}, "
            f"min_dbfs={self.threshold_gate.min_dbfs:+.1f}, "
            f"enhance_threshold_dbfs={self.preprocessor.enhance_threshold_db:+.1f}, "
            f"noise_reduction_db={self.preprocessor.noise_reduction_db:.1f}, "
            f"main_gain_db={self.preprocessor.main_gain_db:+.1f}, "
            f"min_score={self.min_score:.1%}, skip_low_db={self.skip_low_db}"
        )
        print("Ctrl+C로 종료합니다.")

    def run(self) -> None:
        self.print_startup_info()
        with self.microphone:
            while True:
                block = self.audio_queue.pop()
                mono = self.microphone.extract_mono(block)
                self.audio_buffer.append(mono)

                for chunk in self.audio_buffer.pop_ready_chunks():
                    self._process_chunk(chunk)

    def _process_chunk(self, chunk) -> None:  # noqa: ANN001
        timestamp = datetime.now().strftime("%H:%M:%S")
        chunk_dbfs = self.level_meter.calculate_dbfs(chunk)
        over_threshold = self.threshold_gate.is_over_threshold(chunk_dbfs)

        if not over_threshold and self.skip_low_db:
            line = (
                f"[{timestamp}] skip: low_signal | "
                f"level={chunk_dbfs:+.1f} dBFS < {self.threshold_gate.min_dbfs:+.1f} dBFS"
            )
            print(colorize(line, ANSI_RED), flush=True)
            return

        preprocess_result = self.preprocessor.enhance(chunk)

        try:
            prediction = self.inference_engine.predict(preprocess_result.processed_audio)
        except Exception as exc:
            print(f"[{timestamp}] 추론 오류: {exc} | 해당 청크 skip", file=sys.stderr)
            return

        status_reasons = []
        if not over_threshold:
            status_reasons.append(self.threshold_gate.low_signal_message(chunk_dbfs))
        if prediction.best_probability < self.min_score:
            status_reasons.append(
                f"점수낮음 {prediction.best_probability:.1%}<{self.min_score:.1%}"
            )

        if status_reasons:
            status = "낮음(" + ", ".join(status_reasons) + ")"
            line_color = ANSI_RED
        else:
            status = "감지"
            line_color = ANSI_GREEN

        line = (
            f"[{timestamp}] 예측: {prediction.best_label} ({prediction.best_probability:.1%}) | "
            f"status={status} | "
            f"level={chunk_dbfs:+.1f} dBFS | "
            f"enhanced={preprocess_result.enhanced_dbfs:+.1f} dBFS | "
            f"quiet_gain={preprocess_result.quiet_gain:.2f}x "
            f"loud_gain={preprocess_result.loud_gain:.2f}x"
            f"{' clipped' if preprocess_result.clipped else ''} | "
            f"전체: {format_scores(prediction.scores)}"
        )
        print(colorize(line, line_color), flush=True)
