from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import torch
import torchaudio

from realtime_inference import (
    MODEL_SAMPLE_RATE,
    SAMPLE_RATE,
    build_custom_label_indices,
    load_efficientat,
    predict_chunk,
)


class SoundRecognitionModel:
    """EfficientAT 모델 로딩 및 predict_chunk 실행 담당."""

    def __init__(self, efficientat_dir: str | Path, device: torch.device) -> None:
        self.efficientat_dir = Path(efficientat_dir)
        self.device = device
        self.model: torch.nn.Module | None = None
        self.mel: torch.nn.Module | None = None
        self.resampler: torch.nn.Module | None = None
        self.audioset_labels: Sequence[str] = []
        self.custom_indices: Dict[str, List[int]] = {}

    def load(self) -> None:
        if SAMPLE_RATE != MODEL_SAMPLE_RATE:
            self.resampler = torchaudio.transforms.Resample(
                orig_freq=SAMPLE_RATE,
                new_freq=MODEL_SAMPLE_RATE,
            ).to(self.device).eval()
        self.model, self.mel, self.audioset_labels = load_efficientat(self.efficientat_dir, self.device)
        self.custom_indices = build_custom_label_indices(self.audioset_labels)

    def predict(self, chunk, debug: bool = False) -> Tuple[str, float, Dict[str, float]]:
        if self.model is None or self.mel is None:
            raise RuntimeError("SoundRecognitionModel.load() must be called before predict().")
        return predict_chunk(
            chunk,
            self.model,
            self.mel,
            self.resampler,
            self.custom_indices,
            self.audioset_labels,
            self.device,
            debug=debug,
        )
