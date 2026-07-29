"""
MobileNetV4-Small detector (timm, trained).

MobileNetV4 has no audio pretraining, so this detector is TRAINED on the target
group (water / knock / dog / cat) by `train_mobilenetv4.py`. This module provides:

  * MelImageTransform : 16 kHz waveform -> [3, H, W] log-mel "image" (shared by
                        both training and inference so preprocessing matches).
  * MobileNetV4Detector : loads the trained checkpoint and returns per-class
                          sigmoid probabilities (multi-label head, independent
                          per-class judgement).

Runs in the existing torch venv (Env A). Requires `timm` (pip install timm).
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torchaudio

from .. import config
from .base import BaseDetector


class MelImageTransform(nn.Module):
    """Turn a 16 kHz mono waveform batch into a normalized 3-channel mel image.

    Deterministic (no SpecAugment) — used identically at train and inference time.
    Output: float32 tensor [B, 3, img_size, img_size], per-sample min-max scaled.
    """

    def __init__(self, sr: int = config.DATASET_SR, n_mels: int = config.MOBILENETV4_N_MELS,
                 img_size: int = config.MOBILENETV4_IMG_SIZE):
        super().__init__()
        self.img_size = img_size
        self.melspec = torchaudio.transforms.MelSpectrogram(
            sample_rate=sr, n_fft=1024, win_length=400, hop_length=160,
            n_mels=n_mels, f_min=0.0, f_max=sr // 2, power=2.0,
        )
        self.to_db = torchaudio.transforms.AmplitudeToDB(stype="power", top_db=80.0)

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        if wav.dim() == 1:
            wav = wav.unsqueeze(0)
        mel = self.to_db(self.melspec(wav))                 # [B, n_mels, T]
        # per-sample min-max normalize to [0, 1]
        b = mel.shape[0]
        flat = mel.reshape(b, -1)
        mn = flat.min(dim=1, keepdim=True).values.view(b, 1, 1)
        mx = flat.max(dim=1, keepdim=True).values.view(b, 1, 1)
        mel = (mel - mn) / (mx - mn + 1e-6)
        mel = mel.unsqueeze(1)                               # [B, 1, n_mels, T]
        mel = nn.functional.interpolate(
            mel, size=(self.img_size, self.img_size),
            mode="bilinear", align_corners=False,
        )
        return mel.repeat(1, 3, 1, 1)                        # [B, 3, H, W]


def build_backbone(num_classes: int, pretrained: bool = True) -> nn.Module:
    """Create the MobileNetV4-Small classifier (multi-label head)."""
    import timm
    return timm.create_model(
        config.MOBILENETV4_TIMM_NAME, pretrained=pretrained, num_classes=num_classes,
    )


class MobileNetV4Detector(BaseDetector):
    target_classes = list(config.DETECTOR_CLASSES["mobilenetv4"])

    def __init__(self, ckpt_path=None, device: str | None = None, batch_size: int = 64):
        self.device = torch.device(device or config.get_device())
        self.batch_size = batch_size
        ckpt_path = ckpt_path or config.MOBILENETV4_CKPT

        ckpt = torch.load(ckpt_path, map_location="cpu")
        # checkpoint stores the class order it was trained with (authoritative)
        self.target_classes = ckpt.get("target_classes", self.target_classes)

        self.model = build_backbone(len(self.target_classes), pretrained=False)
        self.model.load_state_dict(ckpt["model_state"])
        self.model.to(self.device).eval()

        self.transform = MelImageTransform().to(self.device).eval()
        print(f"[MobileNetV4] loaded {ckpt_path} | classes={self.target_classes}")

    @torch.no_grad()
    def predict_proba_batch(self, waveforms_16k: np.ndarray) -> np.ndarray:
        out = np.zeros((len(waveforms_16k), len(self.target_classes)), dtype=np.float32)
        for start in range(0, len(waveforms_16k), self.batch_size):
            chunk = waveforms_16k[start:start + self.batch_size]
            wav = torch.from_numpy(np.ascontiguousarray(chunk)).to(self.device)
            img = self.transform(wav)
            logits = self.model(img)
            out[start:start + len(chunk)] = torch.sigmoid(logits.float()).cpu().numpy()
        return out
