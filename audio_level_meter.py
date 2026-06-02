from __future__ import annotations

import numpy as np

from realtime_inference import rms_dbfs


class AudioLevelMeter:
    """오디오 chunk의 음압 레벨을 dBFS 단위로 계산."""

    def __init__(self) -> None:
        self.current_dbfs = float("-inf")

    def calculate_dbfs(self, chunk: np.ndarray) -> float:
        self.current_dbfs = float(rms_dbfs(chunk))
        return self.current_dbfs
