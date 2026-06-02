from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from realtime_inference import enhance_chunk, rms_dbfs


@dataclass(frozen=True)
class PreprocessResult:
    audio: np.ndarray
    enhanced_dbfs: float
    clipped: bool
    quiet_gain: float
    loud_gain: float


class AudioPreprocessor:
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

    def preprocess(self, chunk: np.ndarray) -> PreprocessResult:
        audio, clipped, quiet_gain, loud_gain = enhance_chunk(
            chunk,
            self.enhance_threshold_db,
            self.noise_reduction_db,
            self.main_gain_db,
            self.enhance_sharpness,
        )
        return PreprocessResult(
            audio=audio,
            enhanced_dbfs=float(rms_dbfs(audio)),
            clipped=bool(clipped),
            quiet_gain=float(quiet_gain),
            loud_gain=float(loud_gain),
        )
