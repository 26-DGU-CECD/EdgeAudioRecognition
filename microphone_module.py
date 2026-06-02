from __future__ import annotations

import sys
import numpy as np
import realtime_inference_ble as ble

from audio_queue import AudioQueue
from realtime_inference import REQUIRED_INPUT_CHANNELS, SAMPLE_RATE, find_respeaker_device


class MicrophoneModule:
    def __init__(
        self,
        *,
        device_index: int,
        device_info: dict,
        stream_channels: int,
        channel_index: int,
        audio_queue: AudioQueue,
    ) -> None:
        self.device_index = device_index
        self.device_info = device_info
        self.stream_channels = int(stream_channels)
        self.channel_index = int(channel_index)
        self.audio_queue = audio_queue
        self.sample_rate = SAMPLE_RATE
        self._stream = None
        self._validate()

    @classmethod
    def auto_detect(cls, *, device_index: int | None, channel_index: int, audio_queue: AudioQueue) -> "MicrophoneModule":
        found_device_index, device_info, stream_channels = find_respeaker_device(device_index)
        return cls(
            device_index=found_device_index,
            device_info=device_info,
            stream_channels=stream_channels,
            channel_index=channel_index,
            audio_queue=audio_queue,
        )

    def _validate(self) -> None:
        if self.channel_index < 0 or self.channel_index >= self.stream_channels:
            raise RuntimeError(
                f"Selected channel index is out of range: channel={self.channel_index}, "
                f"available=0..{self.stream_channels - 1}"
            )
        if self.stream_channels < REQUIRED_INPUT_CHANNELS:
            print(
                f"Warning: selected input device reports only {self.stream_channels} input channels. "
                "Continuing with the available channel.",
                file=sys.stderr,
                flush=True,
            )

    def _callback(self, indata, frames, time_info, status) -> None:  # noqa: ANN001
        if status:
            print(f"Audio input status: {status}", file=sys.stderr, flush=True)
        self.audio_queue.push(indata.copy())

    def open(self):
        self._stream = ble.sd.InputStream(
            device=self.device_index,
            samplerate=SAMPLE_RATE,
            channels=self.stream_channels,
            dtype="float32",
            callback=self._callback,
        )
        return self._stream

    def select_model_channel(self, chunk_multi: np.ndarray) -> np.ndarray:
        return chunk_multi[:, self.channel_index].astype(np.float32, copy=True)
