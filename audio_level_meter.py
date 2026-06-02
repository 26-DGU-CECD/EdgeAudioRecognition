from __future__ import annotations

import numpy as np
from realtime_inference import rms_dbfs


class AudioLevelMeter:
    def calculate_dbfs(self, chunk: np.ndarray) -> float:
        return float(rms_dbfs(chunk))
