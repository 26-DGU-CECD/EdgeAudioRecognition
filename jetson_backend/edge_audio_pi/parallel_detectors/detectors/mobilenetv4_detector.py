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


def power_to_db_per_sample(spec: torch.Tensor, top_db: float = 80.0,
                           amin: float = 1e-10) -> torch.Tensor:
    """[B, F, T] power spectrogram -> dB, clamped `top_db` below EACH SAMPLE's own peak.

    Numerically identical to ``AmplitudeToDB(stype="power", top_db=top_db)`` evaluated
    one sample at a time, but independent of batch size.

    Why this exists: ``torchaudio.functional.amplitude_to_DB`` sets
    ``packed_channels = shape[-3] if dim > 2 else 1``. For a 3-D ``[B, F, T]`` input
    that makes B the *channel* axis, so it reshapes to ``[1, B, F, T]`` and takes a
    single ``amax`` over the WHOLE BATCH — one shared cutoff for every clip. The same
    clip therefore produced a different image at training (bs 32), offline scoring
    (bs 64) and Pi realtime (bs 1). Measured drift: 28.9 dB in the raw dB domain,
    0.97 on the normalized mel image, 3/400 decision flips.
    See `pi_backend/README.md` §5 and `parallel_detectors/test_preproc_parity.py`.
    """
    db = 10.0 * torch.log10(torch.clamp(spec, min=amin))
    b = db.shape[0]
    cutoff = db.reshape(b, -1).amax(dim=1).view(b, 1, 1) - top_db  # batch-safe: per-sample
    return torch.maximum(db, cutoff)


class MelImageTransform(nn.Module):
    """Turn a 16 kHz mono waveform batch into a normalized 3-channel mel image.

    Deterministic (no SpecAugment) and BATCH-INDEPENDENT — used identically at train
    and inference time, and asserted bitwise-identical across batch sizes by
    `parallel_detectors/test_preproc_parity.py`. SpecAugment belongs in the trainer,
    not here: keeping this module deterministic is what makes train/inference parity
    provable.

    `norm` selects the normalization applied to the dB-scale mel:
      * "minmax"       - per-sample min-max to [0, 1] (historical default). Batch
                         independent, but a per-clip contrast stretch: it discards
                         absolute level, which is a real cue (e.g. water vs knock),
                         and collapses digitally-silent windows (mx - mn == 0).
      * "fixed_affine" - clamp((db - db_floor) / (db_ceil - db_floor), 0, 1). No
                         per-sample statistics at all; preserves absolute level.
                         Same convention as EfficientAT's `(log_mel + 4.5) / 5`.
      * "imagenet"     - minmax to [0, 1], then ImageNet mean/std standardization,
                         which is what the pretrained timm backbone expects.

    Output: float32 tensor [B, 3, img_size, img_size].
    """

    NORMS = ("minmax", "fixed_affine", "imagenet")

    def __init__(self, sr: int = config.DATASET_SR, n_mels: int = config.MOBILENETV4_N_MELS,
                 img_size: int = config.MOBILENETV4_IMG_SIZE, top_db: float = 80.0,
                 norm: str = "minmax", db_floor: float = -80.0, db_ceil: float = 0.0,
                 imagenet_mean=(0.485, 0.456, 0.406), imagenet_std=(0.229, 0.224, 0.225)):
        super().__init__()
        if norm not in self.NORMS:
            raise ValueError(f"norm must be one of {self.NORMS}, got {norm!r}")
        self.img_size = img_size
        self.top_db = float(top_db)
        self.norm = norm
        self.db_floor = float(db_floor)
        self.db_ceil = float(db_ceil)
        self.melspec = torchaudio.transforms.MelSpectrogram(
            sample_rate=sr, n_fft=1024, win_length=400, hop_length=160,
            n_mels=n_mels, f_min=0.0, f_max=sr // 2, power=2.0,
        )
        self.register_buffer("_in_mean", torch.tensor(imagenet_mean).view(1, 3, 1, 1))
        self.register_buffer("_in_std", torch.tensor(imagenet_std).view(1, 3, 1, 1))

    def _normalize(self, mel: torch.Tensor) -> torch.Tensor:
        """[B, F, T] dB-scale mel -> [B, F, T] in roughly [0, 1]. Per-sample only."""
        if self.norm == "fixed_affine":
            span = self.db_ceil - self.db_floor
            return ((mel - self.db_floor) / span).clamp(0.0, 1.0)
        b = mel.shape[0]
        flat = mel.reshape(b, -1)
        mn = flat.amin(dim=1).view(b, 1, 1)                  # batch-safe: per-sample
        mx = flat.amax(dim=1).view(b, 1, 1)                  # batch-safe: per-sample
        return (mel - mn) / (mx - mn + 1e-6)

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        if wav.dim() == 1:
            wav = wav.unsqueeze(0)
        mel = power_to_db_per_sample(self.melspec(wav), self.top_db)   # [B, n_mels, T]
        mel = self._normalize(mel)
        mel = mel.unsqueeze(1)                               # [B, 1, n_mels, T]
        mel = nn.functional.interpolate(
            mel, size=(self.img_size, self.img_size),
            mode="bilinear", align_corners=False,
        )
        img = mel.repeat(1, 3, 1, 1)                         # [B, 3, H, W]
        if self.norm == "imagenet":
            img = (img - self._in_mean) / self._in_std
        return img


def build_backbone(num_classes: int, pretrained: bool = True) -> nn.Module:
    """Create the MobileNetV4-Small classifier (multi-label head)."""
    import timm
    return timm.create_model(
        config.MOBILENETV4_TIMM_NAME, pretrained=pretrained, num_classes=num_classes,
    )


class MobileNetV4Detector(BaseDetector):
    target_classes = list(config.DETECTOR_CLASSES["mobilenetv4"])

    provides_logits = True

    def __init__(self, ckpt_path=None, device: str | None = None, batch_size: int = 64,
                 calibration=None):
        self.device = torch.device(device or config.get_device())
        self.batch_size = batch_size
        ckpt_path = ckpt_path or config.MOBILENETV4_CKPT

        ckpt = torch.load(ckpt_path, map_location="cpu")
        # checkpoint stores the class order it was trained with (authoritative)
        self.target_classes = ckpt.get("target_classes", self.target_classes)

        self.model = build_backbone(len(self.target_classes), pretrained=False)
        self.model.load_state_dict(ckpt["model_state"])
        self.model.to(self.device).eval()

        # The checkpoint carries the preprocessing it was trained with (authoritative).
        # Older checkpoints predate `mel_kwargs` and used the min-max defaults.
        mel_kwargs = ckpt.get("mel_kwargs", {})
        self.transform = MelImageTransform(**mel_kwargs).to(self.device).eval()
        print(f"[MobileNetV4] loaded {ckpt_path} | classes={self.target_classes} "
              f"| norm={self.transform.norm}")
        self._init_calibration("mobilenetv4", calibration, ckpt_path)

    @torch.no_grad()
    def predict_logits_batch(self, waveforms_16k: np.ndarray) -> np.ndarray:
        out = np.zeros((len(waveforms_16k), len(self.target_classes)), dtype=np.float32)
        for start in range(0, len(waveforms_16k), self.batch_size):
            chunk = waveforms_16k[start:start + self.batch_size]
            wav = torch.from_numpy(np.ascontiguousarray(chunk)).to(self.device)
            img = self.transform(wav)
            out[start:start + len(chunk)] = self.model(img).float().cpu().numpy()
        return out

    def predict_proba_batch(self, waveforms_16k: np.ndarray) -> np.ndarray:
        return self.probs_from_logits(self.predict_logits_batch(waveforms_16k))
