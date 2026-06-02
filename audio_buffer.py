from __future__ import annotations

from typing import List

import numpy as np


class AudioBuffer:
    """짧은 audio block들을 모아 고정 길이 chunk로 만드는 버퍼."""

    def __init__(self, chunk_samples: int) -> None:
        self.chunk_samples = int(chunk_samples)
        self.pending_blocks: List[np.ndarray] = []
        self.pending_samples = 0
        self._joined: np.ndarray | None = None
        self._offset = 0

    def append(self, block: np.ndarray) -> None:
        block = block.astype(np.float32, copy=True)
        self.pending_blocks.append(block)
        self.pending_samples += block.shape[0]

    def is_chunk_ready(self) -> bool:
        return self.pending_samples >= self.chunk_samples

    def prepare_joined_blocks(self) -> None:
        self._joined = np.concatenate(self.pending_blocks, axis=0)
        self._offset = 0

    def has_next_chunk(self) -> bool:
        if self._joined is None:
            return False
        return self._joined.shape[0] - self._offset >= self.chunk_samples

    def get_chunk(self) -> np.ndarray:
        if self._joined is None:
            self.prepare_joined_blocks()
        assert self._joined is not None
        chunk = self._joined[self._offset : self._offset + self.chunk_samples]
        self._offset += self.chunk_samples
        return chunk

    def clear_processed(self) -> None:
        if self._joined is None:
            self.pending_blocks = []
            self.pending_samples = 0
            return

        remainder = self._joined[self._offset :]
        self.pending_blocks = [remainder] if remainder.size else []
        self.pending_samples = remainder.shape[0]
        self._joined = None
        self._offset = 0
