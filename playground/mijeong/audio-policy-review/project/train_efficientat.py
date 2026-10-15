from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import torchaudio
from torch import nn

import config
from common_torch import (
    add_common_args,
    audio_config_from_args,
    make_loaders,
    train_model,
)


EFFICIENTAT_WIDTH = {
    "mn04_as": 0.4,
    "mn05_as": 0.5,
    "mn10_as": 1.0,
}


class EfficientATFrontend(nn.Module):
    """EfficientAT AudioSet 설정에 맞춘 deterministic log-mel frontend."""

    def __init__(self, sr: int = 32_000, n_mels: int = 128) -> None:
        super().__init__()

        self.sr = int(sr)
        self.n_fft = 1024
        self.win_length = 800
        self.hop_length = 320

        self.register_buffer(
            "window",
            torch.hann_window(self.win_length, periodic=False),
            persistent=False,
        )
        self.register_buffer(
            "preemphasis",
            torch.tensor([[[-0.97, 1.0]]], dtype=torch.float32),
            persistent=False,
        )

        # EfficientAT eval frontend의 기본 범위: 0 Hz ~ 15 kHz.
        mel_basis, _ = torchaudio.compliance.kaldi.get_mel_banks(
            n_mels,
            self.n_fft,
            self.sr,
            0.0,
            15_000.0,
            vtln_low=100.0,
            vtln_high=-500.0,
            vtln_warp_factor=1.0,
        )
        mel_basis = F.pad(mel_basis, (0, 1), value=0.0)
        self.register_buffer("mel_basis", mel_basis, persistent=False)

    def forward(self, wav_32k: torch.Tensor) -> torch.Tensor:
        """Convert [B, samples] waveform into [B, 128, time] log-mel."""
        if wav_32k.ndim != 2:
            raise ValueError(
                "EfficientATFrontend input must have shape [batch, samples], "
                f"but received {tuple(wav_32k.shape)}"
            )

        # Pre-emphasis: [B, T] -> [B, T-1]
        x = F.conv1d(
            wav_32k.unsqueeze(1),
            self.preemphasis.to(dtype=wav_32k.dtype),
        ).squeeze(1)

        stft = torch.stft(
            x,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window.to(dtype=x.dtype),
            center=True,
            normalized=False,
            return_complex=True,
        )
        power = stft.abs().square()

        # [128, 513] @ [B, 513, T] -> [B, 128, T]
        mel = torch.matmul(
            self.mel_basis.to(device=power.device, dtype=power.dtype),
            power,
        )
        mel = torch.log(mel + 1e-5)

        # EfficientAT에서 사용하는 fast normalization.
        return (mel + 4.5) / 5.0


class EfficientATBaseline(nn.Module):
    def __init__(
        self,
        efficientat_root: str | Path,
        model_name: str,
        num_classes: int,
        freeze_backbone: bool = True,
    ) -> None:
        super().__init__()

        root = Path(efficientat_root).expanduser().resolve()
        if not root.is_dir():
            raise FileNotFoundError(f"EfficientAT repository not found: {root}")

        metadata_file = root / "metadata" / "class_labels_indices.csv"
        if not metadata_file.is_file():
            raise FileNotFoundError(
                "EfficientAT metadata file not found: "
                f"{metadata_file}\n"
                "Clone the complete EfficientAT repository again."
            )

        if model_name not in EFFICIENTAT_WIDTH:
            raise ValueError(
                f"Unsupported EfficientAT model: {model_name}. "
                f"Choose one of {sorted(EFFICIENTAT_WIDTH)}"
            )

        root_str = str(root)
        if root_str not in sys.path:
            sys.path.insert(0, root_str)

        # EfficientAT 내부에서 metadata/... 및 resources/... 상대경로를 사용한다.
        # import와 사전학습 모델 생성 동안에만 현재 작업 폴더를 저장소 루트로 바꾼다.
        original_cwd = Path.cwd()
        try:
            os.chdir(root)
            from models.mn.model import get_model as get_mn  # type: ignore

            base = get_mn(
                num_classes=527,
                pretrained_name=model_name,
                width_mult=EFFICIENTAT_WIDTH[model_name],
                head_type="mlp",
                input_dim_f=128,
                input_dim_t=201,
            )
        finally:
            os.chdir(original_cwd)

        # 사전학습 feature extractor는 유지하고 14-class용 256-unit head로 교체한다.
        feature_dim = int(base.classifier[2].in_features)
        base.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(start_dim=1),
            nn.Linear(feature_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.30),
            nn.Linear(256, int(num_classes)),
        )

        self.base = base
        self.frontend = EfficientATFrontend(sr=32_000, n_mels=128)
        self.freeze_backbone = bool(freeze_backbone)

        if self.freeze_backbone:
            for parameter in self.base.features.parameters():
                parameter.requires_grad = False
            self.base.features.eval()

    def train(self, mode: bool = True):
        """Keep the frozen backbone's BatchNorm statistics fixed during training."""
        super().train(mode)
        if self.freeze_backbone:
            self.base.features.eval()
        return self

    def forward(self, wav_16k: torch.Tensor) -> torch.Tensor:
        """Forward 16 kHz waveform [B, 32000] and return logits [B, classes]."""
        if wav_16k.ndim != 2:
            raise ValueError(
                "EfficientATBaseline input must have shape [batch, samples], "
                f"but received {tuple(wav_16k.shape)}"
            )

        # STFT/log-mel은 항상 float32로 계산한다. 외부 AMP가 활성화되어 있어도
        # frontend에서 half-precision STFT 오류가 나지 않게 한다.
        with torch.autocast(device_type=wav_16k.device.type, enabled=False):
            # 공통 파이프라인 출력은 16 kHz이고 EfficientAT frontend는 32 kHz를 사용한다.
            wav_32k = torchaudio.functional.resample(
                wav_16k.float(),
                orig_freq=16_000,
                new_freq=32_000,
            )

            # [B, 128, T] -> [B, 1, 128, T]
            mel = self.frontend(wav_32k).unsqueeze(1)

        # EfficientAT MN.forward() returns (logits, pooled_features).
        output = self.base(mel)
        logits = output[0] if isinstance(output, tuple) else output

        if logits.ndim != 2:
            raise RuntimeError(
                "EfficientAT logits must have shape [batch, classes], "
                f"but received {tuple(logits.shape)}"
            )
        return logits


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train the frozen-backbone EfficientAT baseline."
    )
    add_common_args(parser, config.EFFICIENTAT_OUTPUT)
    parser.add_argument(
        "--efficientat-root",
        default=str(config.EFFICIENTAT_ROOT),
        help="Path to the cloned official EfficientAT repository.",
    )
    parser.add_argument(
        "--model-name",
        choices=sorted(EFFICIENTAT_WIDTH),
        default=config.EFFICIENTAT_MODEL_NAME,
    )
    args = parser.parse_args()

    config.validate_paths(require_efficientat=True)
    torch.manual_seed(args.seed)

    class_names, train_loader, val_loader, test_loader = make_loaders(
        args.csv,
        args.classes,
        audio_config_from_args(args),
        args.batch_size,
        args.num_workers,
        args.seed,
    )

    model = EfficientATBaseline(
        efficientat_root=args.efficientat_root,
        model_name=args.model_name,
        num_classes=len(class_names),
        freeze_backbone=True,
    )

    train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        class_names=class_names,
        output_dir=args.output,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        patience=args.patience,
        device=args.device,
    )


if __name__ == "__main__":
    main()
