from __future__ import annotations

from typing import List, Tuple
import numpy as np


class AudioBuffer:
    def __init__(self, chunk_samples: int) -> None:
        self.chunk_samples = int(chunk_samples)
        self.pending_blocks: List[np.ndarray] = []
        self.pending_samples = 0

    def append(self, block: np.ndarray) -> None:
        block = block.astype(np.float32, copy=True)
        self.pending_blocks.append(block)
        self.pending_samples += block.shape[0]

    def is_chunk_ready(self) -> bool:
        return self.pending_samples >= self.chunk_samples

    def pop_ready_chunks(self) -> List[np.ndarray]:
        if not self.is_chunk_ready():
            return []
        joined = np.concatenate(self.pending_blocks, axis=0)
        chunks: List[np.ndarray] = []
        offset = 0
        while joined.shape[0] - offset >= self.chunk_samples:
            chunks.append(joined[offset : offset + self.chunk_samples])
            offset += self.chunk_samples
        remainder = joined[offset:]
        self.pending_blocks = [remainder] if remainder.size else []
        self.pending_samples = remainder.shape[0]
        return chunks
