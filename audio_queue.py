from __future__ import annotations

import queue

import numpy as np


class AudioQueue:
    """Audio input callback에서 들어온 block을 안전하게 전달하는 큐."""

    def __init__(self) -> None:
        self._queue: "queue.Queue[np.ndarray]" = queue.Queue()

    def push(self, block: np.ndarray) -> None:
        self._queue.put(block.copy())

    def pop(self) -> np.ndarray:
        return self._queue.get()

    def is_empty(self) -> bool:
        return self._queue.empty()
