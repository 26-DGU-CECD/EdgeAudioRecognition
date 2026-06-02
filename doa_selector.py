from __future__ import annotations

import numpy as np

from doa_audio_estimator import AudioDOAEstimator
from doa_reading import DOAReading
from doa_usb_reader import DOAReader


class DOASelector:
    def __init__(self, *, doa_source: str, usb_reader: DOAReader, audio_estimator: AudioDOAEstimator) -> None:
        self.doa_source = doa_source
        self.usb_reader = usb_reader
        self.audio_estimator = audio_estimator

    def choose(self, chunk: np.ndarray) -> DOAReading:
        if self.doa_source == "audio":
            return self.audio_estimator.estimate(chunk)
        if self.doa_source == "usb":
            return self.usb_reader.snapshot()

        audio_reading = self.audio_estimator.estimate(chunk)
        if audio_reading.raw_angle is not None:
            return audio_reading

        usb_reading = self.usb_reader.snapshot()
        if usb_reading.raw_angle is not None:
            return usb_reading

        return DOAReading(None, "none", f"{audio_reading.status};{usb_reading.status}")
