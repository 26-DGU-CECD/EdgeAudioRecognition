"""
EfficientAT detector (mn10_as). Two modes:

  * zero-shot : read sigmoid probs of the AudioSet classes that map to the target
                classes (bicycle_bell/baby_cry/gunshot/glass_break), MAX over
                candidates. Clips are TILED 2 s -> 10 s (mn10_as saturates on short
                inputs).
  * trained   : mn10_as backbone + a fine-tuned 4-class head (see train_efficientat.py).
                Fed 2 s clips directly (the head was trained at that length).

`run_inference.py` picks trained mode automatically when a checkpoint exists.
Runs in the `paradet` env (torch). Input clips are 16 kHz; mn10_as wants 32 kHz,
so waveforms are resampled 16k -> 32k before the mel transform.
"""
from __future__ import annotations

import csv
import os
import sys

import numpy as np
import torch
from torchaudio.functional import resample

from .. import config
from .base import BaseDetector


def _import_efficientat():
    """Import EfficientAT modules with the repo root as cwd (relative CSV loads)."""
    repo = str(config.EFFICIENTAT_DIR)
    if repo not in sys.path:
        sys.path.insert(0, repo)
    prev_cwd = os.getcwd()
    os.chdir(repo)
    try:
        from models.mn.model import get_model
        from models.preprocess import AugmentMelSTFT
        from helpers.utils import NAME_TO_WIDTH
    finally:
        os.chdir(prev_cwd)
    return get_model, AugmentMelSTFT, NAME_TO_WIDTH


def build_efficientat(num_classes: int, pretrained: bool = True):
    """Build an mn10_as model. num_classes=527 keeps the AudioSet head;
    num_classes<527 with pretrained=True drops it for fine-tuning."""
    get_model, _, NAME_TO_WIDTH = _import_efficientat()
    name = config.EFFICIENTAT_MODEL_NAME
    return get_model(
        width_mult=NAME_TO_WIDTH(name),
        pretrained_name=name if pretrained else None,
        num_classes=num_classes,
        head_type="mlp",
    )


def build_mel(device):
    _, AugmentMelSTFT, _ = _import_efficientat()
    return AugmentMelSTFT(
        n_mels=128, sr=config.EFFICIENTAT_SR, win_length=800, hopsize=320,
    ).to(device)


def _load_audioset_display_names(csv_path) -> list[str]:
    rows = {}
    with open(csv_path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows[int(r["index"])] = r["display_name"]
    return [rows[i] for i in range(len(rows))]


class EfficientATDetector(BaseDetector):
    target_classes = list(config.DETECTOR_CLASSES["efficientat"])

    provides_logits = True

    def __init__(self, device: str | None = None, batch_size: int = 64, ckpt=None,
                 calibration=None):
        self.device = torch.device(device or config.get_device())
        self.batch_size = batch_size
        self.mel = build_mel(self.device).eval()
        self.target_len = int(config.EFFICIENTAT_TARGET_SECONDS * config.EFFICIENTAT_SR)

        if ckpt is not None:
            # ---- trained mode ----
            self.mode = "trained"
            state = torch.load(ckpt, map_location="cpu")
            self.target_classes = state["target_classes"]
            self.model = build_efficientat(len(self.target_classes), pretrained=False)
            self.model.load_state_dict(state["model_state"])
            self.model.to(self.device).eval()
            print(f"[EfficientAT] trained head loaded: {ckpt} | {self.target_classes}")
        else:
            # ---- zero-shot mode ----
            self.mode = "zeroshot"
            self.model = build_efficientat(527, pretrained=True).to(self.device).eval()
            names = _load_audioset_display_names(config.EFFICIENTAT_LABELS_CSV)
            name_to_idx = {n: i for i, n in enumerate(names)}
            self.class_indices = {}
            for cls in self.target_classes:
                idxs = [name_to_idx[dn] for dn in config.EFFICIENTAT_AUDIOSET_NAMES[cls]
                        if dn in name_to_idx]
                if not idxs:
                    raise ValueError(f"No AudioSet indices resolved for class {cls!r}")
                self.class_indices[cls] = idxs
            print(f"[EfficientAT] zero-shot class->indices: {self.class_indices}")

        self._init_calibration("efficientat", calibration,
                               ckpt if self.mode == "trained" else None)

    @torch.no_grad()
    def predict_logits_batch(self, waveforms_16k: np.ndarray) -> np.ndarray:
        """Raw logits per target class.

        Zero-shot logits are recoverable despite the max-over-AudioSet-indices step:
        sigmoid is strictly increasing, so max_j sigmoid(z_j) == sigmoid(max_j z_j).
        Taking the max in logit space is therefore exactly the old computation.
        """
        out = np.zeros((len(waveforms_16k), len(self.target_classes)), dtype=np.float32)
        for start in range(0, len(waveforms_16k), self.batch_size):
            chunk = waveforms_16k[start:start + self.batch_size]
            wav = torch.from_numpy(np.ascontiguousarray(chunk)).to(self.device)  # [b, 32000] @16k
            wav = resample(wav, config.DATASET_SR, config.EFFICIENTAT_SR)        # -> @32k
            if self.mode == "zeroshot" and wav.shape[1] < self.target_len:
                reps = -(-self.target_len // wav.shape[1])  # ceil; tile 2s -> 10s
                wav = wav.repeat(1, reps)[:, :self.target_len]
            spec = self.mel(wav)                       # [b, 128, frames]
            logits, _ = self.model(spec.unsqueeze(1))  # [b, n]
            logits = logits.float().cpu().numpy()
            if self.mode == "trained":
                out[start:start + len(chunk)] = logits
            else:
                for j, cls in enumerate(self.target_classes):
                    out[start:start + len(chunk), j] = \
                        logits[:, self.class_indices[cls]].max(axis=1)
        return out

    def predict_proba_batch(self, waveforms_16k: np.ndarray) -> np.ndarray:
        return self.probs_from_logits(self.predict_logits_batch(waveforms_16k))
