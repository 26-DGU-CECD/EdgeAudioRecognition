from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from realtime_inference import enhance_chunk, rms_dbfs


@dataclass(frozen=True)
class PreprocessResult:
    original_dbfs: float
    enhanced_dbfs: float
    is_clipped: bool
    quiet_gain: float
    loud_gain: float
    processed_audio: np.ndarray


class AudioPreprocessor:
    """작은 신호는 줄이고 큰 신호는 강조한 뒤 모델 입력용 오디오로 변환."""

    def __init__(
        self,
        *,
        enhance_threshold_db: float,
        noise_reduction_db: float,
        main_gain_db: float,
        enhance_sharpness: float,
    ) -> None:
        self.enhance_threshold_db = float(enhance_threshold_db)
        self.noise_reduction_db = float(noise_reduction_db)
        self.main_gain_db = float(main_gain_db)
        self.enhance_sharpness = float(enhance_sharpness)

    def enhance(self, chunk: np.ndarray) -> PreprocessResult:
        original_dbfs = float(rms_dbfs(chunk))
        processed_audio, clipped, quiet_gain, loud_gain = enhance_chunk(
            chunk,
            self.enhance_threshold_db,
            self.noise_reduction_db,
            self.main_gain_db,
            self.enhance_sharpness,
        )
        enhanced_dbfs = float(rms_dbfs(processed_audio))
        return PreprocessResult(
            original_dbfs=original_dbfs,
            enhanced_dbfs=enhanced_dbfs,
            is_clipped=bool(clipped),
            quiet_gain=float(quiet_gain),
            loud_gain=float(loud_gain),
            processed_audio=processed_audio,
        )

    def adjust_length(self, chunk: np.ndarray) -> np.ndarray:
        # 기존 코드에서는 predict_chunk 내부에서 모델 입력 길이/샘플링 처리를 수행한다.
        # 필요하면 이 메소드에 pad/crop 로직을 추가하면 된다.
        return chunk

    def preprocess(self, chunk: np.ndarray) -> PreprocessResult:
        result = self.enhance(chunk)
        adjusted = self.adjust_length(result.processed_audio)
        return PreprocessResult(
            original_dbfs=result.original_dbfs,
            enhanced_dbfs=float(rms_dbfs(adjusted)),
            is_clipped=result.is_clipped,
            quiet_gain=result.quiet_gain,
            loud_gain=result.loud_gain,
            processed_audio=adjusted,
        )
