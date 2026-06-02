from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Sequence

import torch
import torchaudio
import numpy as np

from realtime_inference import (
    MODEL_SAMPLE_RATE,
    SAMPLE_RATE,
    build_custom_label_indices,
    load_efficientat,
    predict_chunk,
)


class ModelInferenceEngine:
    def __init__(self, efficientat_dir: str | Path, *, debug: bool = False) -> None:
        self.efficientat_dir = Path(efficientat_dir)
        self.debug = bool(debug)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.resampler: torch.nn.Module | None = None
        self.model: torch.nn.Module | None = None
        self.mel: torch.nn.Module | None = None
        self.audioset_labels: Sequence[str] = []
        self.custom_indices: Dict[str, List[int]] = {}

    def load(self) -> None:
        print(f"Inference device: {self.device}")
        if SAMPLE_RATE != MODEL_SAMPLE_RATE:
            self.resampler = torchaudio.transforms.Resample(
                orig_freq=SAMPLE_RATE,
                new_freq=MODEL_SAMPLE_RATE,
            ).to(self.device).eval()
        self.model, self.mel, self.audioset_labels = load_efficientat(self.efficientat_dir, self.device)
        self.custom_indices = build_custom_label_indices(self.audioset_labels)

    def predict(self, audio: np.ndarray):
        if self.model is None or self.mel is None:
            raise RuntimeError("ModelInferenceEngine.load() must be called before predict().")
        return predict_chunk(
            audio,
            self.model,
            self.mel,
            self.resampler,
            self.custom_indices,
            self.audioset_labels,
            self.device,
            debug=self.debug,
        )
