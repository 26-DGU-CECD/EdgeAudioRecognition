from __future__ import annotations

from dataclasses import dataclass
from math import gcd
from pathlib import Path
from typing import Tuple

import numpy as np
import soundfile as sf
from scipy.signal import butter, resample_poly, sosfilt, sosfilt_zi


@dataclass(frozen=True)
class AudioConfig:
    sample_rate: int = 16_000
    clip_seconds: float = 2.0
    highpass_hz: float = 80.0  # 배포 시 확정한 값으로 변경
    highpass_order: int = 2
    silence_frame_ms: float = 100.0
    silence_threshold_dbfs: float = -30.0
    target_rms_dbfs: float = -20.0  # 배포 시 확정한 루드니스 목표값으로 변경
    max_normalize_gain_db: float = 20.0

    @property
    def num_samples(self) -> int:
        return int(round(self.sample_rate * self.clip_seconds))


def _to_mono(x: np.ndarray) -> np.ndarray:
    if x.ndim == 1:
        return x
    return np.mean(x, axis=1)


def _resample(x: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    if src_sr == dst_sr:
        return x
    factor = gcd(src_sr, dst_sr)
    return resample_poly(x, dst_sr // factor, src_sr // factor)


def _fix_length(x: np.ndarray, length: int) -> np.ndarray:
    if len(x) >= length:
        return x[:length]
    return np.pad(x, (0, length - len(x)), mode="constant")


def highpass_filter(x: np.ndarray, cfg: AudioConfig) -> np.ndarray:
    if cfg.highpass_hz <= 0:
        return x
    nyquist = cfg.sample_rate / 2.0
    if not 0 < cfg.highpass_hz < nyquist:
        raise ValueError(f"highpass_hz must be in (0, {nyquist}), got {cfg.highpass_hz}")
    sos = butter(
        cfg.highpass_order,
        cfg.highpass_hz / nyquist,
        btype="highpass",
        output="sos",
    )
    # causal IIR. 실제 스트리밍 구현에서도 같은 계수/상태를 사용해야 한다.
    zi = sosfilt_zi(sos) * float(x[0] if len(x) else 0.0)
    y, _ = sosfilt(sos, x, zi=zi)
    return y


def max_frame_rms_dbfs(x: np.ndarray, cfg: AudioConfig) -> float:
    frame = max(1, int(round(cfg.sample_rate * cfg.silence_frame_ms / 1000.0)))
    if len(x) == 0:
        return float("-inf")
    values = []
    for start in range(0, len(x), frame):
        chunk = x[start : start + frame]
        if len(chunk) == 0:
            continue
        rms = float(np.sqrt(np.mean(np.square(chunk, dtype=np.float64)) + 1e-12))
        values.append(20.0 * np.log10(max(rms, 1e-12)))
    return max(values) if values else float("-inf")


def rms_normalize(x: np.ndarray, cfg: AudioConfig) -> np.ndarray:
    rms = float(np.sqrt(np.mean(np.square(x, dtype=np.float64)) + 1e-12))
    current_db = 20.0 * np.log10(max(rms, 1e-12))
    gain_db = np.clip(
        cfg.target_rms_dbfs - current_db,
        -cfg.max_normalize_gain_db,
        cfg.max_normalize_gain_db,
    )
    gain = 10.0 ** (gain_db / 20.0)
    return np.clip(x * gain, -1.0, 1.0)


def load_and_preprocess(path: str | Path, cfg: AudioConfig) -> Tuple[np.ndarray, bool, float]:
    """Returns waveform[32000], is_silent, max_100ms_rms_dbfs."""
    x, sr = sf.read(str(path), dtype="float32", always_2d=True)
    x = _to_mono(x).astype(np.float32, copy=False)
    x = _resample(x, int(sr), cfg.sample_rate).astype(np.float32, copy=False)
    x = _fix_length(x, cfg.num_samples)
    x = highpass_filter(x, cfg).astype(np.float32, copy=False)

    max_dbfs = max_frame_rms_dbfs(x, cfg)
    is_silent = max_dbfs < cfg.silence_threshold_dbfs

    # 무음은 정규화하지 않는다. 작은 잡음을 목표 레벨까지 증폭하는 것을 방지한다.
    if not is_silent:
        x = rms_normalize(x, cfg).astype(np.float32, copy=False)

    return np.ascontiguousarray(x, dtype=np.float32), is_silent, max_dbfs
