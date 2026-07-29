from __future__ import annotations

import threading
import time
from collections import deque

import numpy as np


class AudioQueue:
    """Non-blocking producer queue bounded by the actual number of samples."""

    def __init__(self, max_seconds: float, sample_rate: int) -> None:
        self.max_samples = max(1, int(max_seconds * sample_rate))
        self._blocks: deque[np.ndarray] = deque()
        self._queued_samples = 0
        self._condition = threading.Condition()
        self.dropped_blocks = 0
        self.total_dropped_blocks = 0

    @staticmethod
    def _sample_count(block: np.ndarray) -> int:
        return int(block.shape[0])

    def push(self, block: np.ndarray) -> None:
        payload = np.asarray(block).copy()
        if self._sample_count(payload) > self.max_samples:
            payload = payload[-self.max_samples :].copy()

        with self._condition:
            while self._blocks and (
                self._queued_samples + self._sample_count(payload) > self.max_samples
            ):
                dropped = self._blocks.popleft()
                self._queued_samples -= self._sample_count(dropped)
                self.dropped_blocks += 1
                self.total_dropped_blocks += 1
            self._blocks.append(payload)
            self._queued_samples += self._sample_count(payload)
            self._condition.notify()

    def pop(self, timeout: float | None = None) -> np.ndarray | None:
        deadline = None if timeout is None else time.monotonic() + timeout
        with self._condition:
            while not self._blocks:
                if deadline is None:
                    self._condition.wait()
                    continue
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return None
                self._condition.wait(remaining)
            block = self._blocks.popleft()
            self._queued_samples -= self._sample_count(block)
            return block

    def take_dropped(self) -> int:
        with self._condition:
            dropped = self.dropped_blocks
            self.dropped_blocks = 0
            return dropped

    def qsize(self) -> int:
        with self._condition:
            return len(self._blocks)
