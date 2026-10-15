from __future__ import annotations

import argparse

import timm
import torch
import torch.nn.functional as F
import torchaudio
from torch import nn

from common_torch import (
    add_common_args,
    audio_config_from_args,
    make_loaders,
    train_model,
)
import config


class LogMelImage(nn.Module):
    def __init__(self):
        super().__init__()
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=16_000,
            n_fft=1024,
            win_length=400,
            hop_length=160,
            f_min=20.0,
            f_max=8_000.0,
            n_mels=128,
            power=2.0,
        )
        self.register_buffer(
            "mean",
            torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "std",
            torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1),
            persistent=False,
        )

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        x = torch.log(self.mel(wav) + 1e-6)
        mu = x.mean(dim=(1, 2), keepdim=True)
        sigma = x.std(dim=(1, 2), keepdim=True).clamp_min(1e-5)
        x = ((x - mu) / sigma).clamp(-3.0, 3.0)
        x = (x + 3.0) / 6.0
        x = x.unsqueeze(1).repeat(1, 3, 1, 1)
        x = F.interpolate(
            x,
            size=(224, 224),
            mode="bilinear",
            align_corners=False,
        )
        return (x - self.mean) / self.std


class MobileNetV4Baseline(nn.Module):
    def __init__(
        self,
        num_classes: int,
        freeze_backbone: bool = True,
    ):
        super().__init__()
        self.frontend = LogMelImage()
        self.backbone = timm.create_model(
            "mobilenetv4_conv_small.e2400_r224_in1k",
            pretrained=True,
            num_classes=0,
            global_pool="avg",
        )
        self.freeze_backbone = bool(freeze_backbone)

        self.backbone.eval()
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 224, 224)
            feature_dim = int(self.backbone(dummy).shape[-1])
        print(f"MobileNetV4 feature dimension: {feature_dim}")

        self.head = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.30),
            nn.Linear(256, num_classes),
        )
        if self.freeze_backbone:
            for parameter in self.backbone.parameters():
                parameter.requires_grad = False
            self.backbone.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_backbone:
            self.backbone.eval()
        return self

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        image = self.frontend(wav)
        if self.freeze_backbone:
            with torch.no_grad():
                features = self.backbone(image)
        else:
            features = self.backbone(image)
        return self.head(features)


def main():
    parser = argparse.ArgumentParser()
    add_common_args(parser, config.MOBILENETV4_OUTPUT)
    args = parser.parse_args()
    config.validate_paths()

    torch.manual_seed(args.seed)
    class_names, train_loader, val_loader, test_loader = make_loaders(
        args.csv,
        args.classes,
        audio_config_from_args(args),
        args.batch_size,
        args.num_workers,
        args.seed,
    )
    model = MobileNetV4Baseline(
        len(class_names),
        freeze_backbone=True,
    )
    train_model(
        model,
        train_loader,
        val_loader,
        test_loader,
        class_names,
        args.output,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        patience=args.patience,
        device=args.device,
    )


if __name__ == "__main__":
    main()
