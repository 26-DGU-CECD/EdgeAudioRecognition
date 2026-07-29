from __future__ import annotations

import numpy as np


class SlidingWindowBuffer:
    def __init__(self, window_samples: int, hop_samples: int) -> None:
        if window_samples <= 0:
            raise ValueError("window_samples must be positive")
        if hop_samples <= 0:
            raise ValueError("hop_samples must be positive")
        if hop_samples > window_samples:
            raise ValueError(
                f"hop ({hop_samples}) must not exceed window ({window_samples})"
            )
        self.window_samples = int(window_samples)
        self.hop_samples = int(hop_samples)
        self._buffer = np.zeros(0, dtype=np.float32)

    def append(self, mono_block: np.ndarray) -> None:
        block = np.asarray(mono_block, dtype=np.float32).reshape(-1)
        if block.size:
            self._buffer = (
                np.concatenate((self._buffer, block))
                if self._buffer.size
                else block.copy()
            )

    def pop_windows(self) -> list[np.ndarray]:
        windows: list[np.ndarray] = []
        while self._buffer.shape[0] >= self.window_samples:
            windows.append(self._buffer[: self.window_samples].copy())
            self._buffer = self._buffer[self.hop_samples :]
        return windows

    @property
    def pending_samples(self) -> int:
        return int(self._buffer.shape[0])
