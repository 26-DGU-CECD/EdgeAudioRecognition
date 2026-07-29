"""
Common detector interface.

Every detector exposes the SAME minimal contract so `run_inference.py` can drive
any of them without special-casing:

    detector.target_classes            -> list[str]  (canonical class names, ordered)
    detector.predict_proba_batch(wavs) -> np.ndarray [N, len(target_classes)]

`wavs` is a float32 numpy array of shape [N, CLIP_SAMPLES] at 16 kHz mono,
amplitude in [-1, 1]. Each detector internally resamples/preprocesses as needed
and returns independent sigmoid probabilities in [0, 1] for its own classes.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class BaseDetector(ABC):
    #: canonical class names this detector is responsible for, in output order
    target_classes: list[str]

    @abstractmethod
    def predict_proba_batch(self, waveforms_16k: np.ndarray) -> np.ndarray:
        """Return [N, len(target_classes)] sigmoid probabilities for a batch.

        Args:
            waveforms_16k: float32 array [N, CLIP_SAMPLES] @ 16 kHz mono, [-1, 1].
        """
        raise NotImplementedError
