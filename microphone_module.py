from __future__ import annotations

from typing import Any, Callable

import realtime_inference_ble as ble


class MicrophoneModule:
    """sounddevice InputStream을 감싼 마이크 모듈 객체."""

    def __init__(
        self,
        *,
        device_index: int,
        sample_rate: int,
        channels: int,
        dtype: str = "float32",
    ) -> None:
        self.device_index = device_index
        self.sample_rate = sample_rate
        self.channels = channels
        self.dtype = dtype
        self._stream: Any | None = None

    def open(self, callback: Callable[..., None]) -> None:
        self._stream = ble.sd.InputStream(
            device=self.device_index,
            samplerate=self.sample_rate,
            channels=self.channels,
            dtype=self.dtype,
            callback=callback,
        )
        self._stream.__enter__()

    def close(self) -> None:
        if self._stream is not None:
            self._stream.__exit__(None, None, None)
            self._stream = None
