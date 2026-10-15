from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch import nn
from torch.utils.data import BatchSampler, DataLoader, Dataset

from common_audio import AudioConfig, load_and_preprocess
from common_metadata import load_metadata, read_class_names
from evaluation_utils import save_classification_artifacts
import config


class WaveformDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        class_names: Sequence[str],
        audio_cfg: AudioConfig,
    ):
        self.df = df.reset_index(drop=True)
        self.class_names = list(class_names)
        self.label_to_id = {name: i for i, name in enumerate(self.class_names)}
        unknown = sorted(set(self.df["label"].astype(str)) - set(self.class_names))
        if unknown:
            raise ValueError(f"Unknown labels in metadata: {unknown}")
        self.labels = [self.label_to_id[str(v)] for v in self.df["label"]]
        self.group_ids = [str(v) for v in self.df["group_id"]]
        self.audio_cfg = audio_cfg

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        wav, is_silent, max_dbfs = load_and_preprocess(row["path"], self.audio_cfg)
        return (
            torch.from_numpy(wav),
            torch.tensor(self.labels[idx], dtype=torch.long),
            str(row["path"]),
            torch.tensor(is_silent, dtype=torch.bool),
            torch.tensor(max_dbfs, dtype=torch.float32),
        )


class BackgroundFractionBatchSampler(BatchSampler):
    """Exact background fraction with source-group-balanced sampling."""

    def __init__(
        self,
        labels: Sequence[int],
        group_ids: Sequence[str],
        background_id: int,
        batch_size: int,
        background_fraction: float = 0.20,
        seed: int = 42,
    ):
        if not 0.0 < background_fraction < 1.0:
            raise ValueError("background_fraction must be in (0, 1)")
        if len(labels) != len(group_ids):
            raise ValueError("labels and group_ids must have the same length")
        self.labels = list(labels)
        self.group_ids = [str(g) for g in group_ids]
        self.background_id = int(background_id)
        self.batch_size = int(batch_size)
        self.n_bg = int(round(self.batch_size * background_fraction))
        self.n_fg = self.batch_size - self.n_bg
        if self.n_bg <= 0 or self.n_fg <= 0:
            raise ValueError("batch_size is too small")

        self.bg_group_to_indices: dict[str, list[int]] = {}
        self.fg_group_to_indices: dict[str, list[int]] = {}
        for i, (label, group) in enumerate(zip(self.labels, self.group_ids)):
            target = (
                self.bg_group_to_indices
                if label == self.background_id
                else self.fg_group_to_indices
            )
            target.setdefault(group, []).append(i)
        if not self.bg_group_to_indices:
            raise ValueError("No background samples found")
        if not self.fg_group_to_indices:
            raise ValueError("No foreground samples found")
        self.seed = int(seed)
        self.epoch = 0

    def __len__(self) -> int:
        return math.ceil(len(self.fg_group_to_indices) / self.n_fg)

    @staticmethod
    def _group_cycle(
        groups: list[str],
        n: int,
        rng: random.Random,
    ) -> list[str]:
        out: list[str] = []
        while len(out) < n:
            block = groups.copy()
            rng.shuffle(block)
            out.extend(block)
        return out[:n]

    def __iter__(self) -> Iterator[list[int]]:
        rng = random.Random(self.seed + self.epoch)
        self.epoch += 1
        n_batches = len(self)
        fg_groups = self._group_cycle(
            list(self.fg_group_to_indices),
            n_batches * self.n_fg,
            rng,
        )
        bg_groups = self._group_cycle(
            list(self.bg_group_to_indices),
            n_batches * self.n_bg,
            rng,
        )

        for batch_idx in range(n_batches):
            fg_slice = fg_groups[
                batch_idx * self.n_fg : (batch_idx + 1) * self.n_fg
            ]
            bg_slice = bg_groups[
                batch_idx * self.n_bg : (batch_idx + 1) * self.n_bg
            ]
            batch = [rng.choice(self.fg_group_to_indices[g]) for g in fg_slice]
            batch += [rng.choice(self.bg_group_to_indices[g]) for g in bg_slice]
            rng.shuffle(batch)
            yield batch


def make_loaders(
    csv_path: str,
    classes_path: str,
    audio_cfg: AudioConfig,
    batch_size: int = 40,
    num_workers: int = 4,
    seed: int = 42,
):
    class_names = read_class_names(classes_path)
    bg_id = class_names.index("background")
    train_ds = WaveformDataset(
        load_metadata(csv_path, "train"),
        class_names,
        audio_cfg,
    )
    val_ds = WaveformDataset(
        load_metadata(csv_path, "val"),
        class_names,
        audio_cfg,
    )
    test_ds = WaveformDataset(
        load_metadata(csv_path, "test"),
        class_names,
        audio_cfg,
    )

    train_sampler = BackgroundFractionBatchSampler(
        train_ds.labels,
        train_ds.group_ids,
        background_id=bg_id,
        batch_size=batch_size,
        background_fraction=0.20,
        seed=seed,
    )
    loader_kwargs = {
        "num_workers": num_workers,
        "pin_memory": True,
        "persistent_workers": num_workers > 0,
    }
    train_loader = DataLoader(
        train_ds,
        batch_sampler=train_sampler,
        **loader_kwargs,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        **loader_kwargs,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        **loader_kwargs,
    )
    return class_names, train_loader, val_loader, test_loader


def make_eval_loaders(
    csv_path: str,
    classes_path: str,
    audio_cfg: AudioConfig,
    batch_size: int = 40,
    num_workers: int = 4,
):
    class_names = read_class_names(classes_path)
    loader_kwargs = {
        "batch_size": batch_size,
        "shuffle": False,
        "num_workers": num_workers,
        "pin_memory": True,
        "persistent_workers": num_workers > 0,
    }
    val_loader = DataLoader(
        WaveformDataset(load_metadata(csv_path, "val"), class_names, audio_cfg),
        **loader_kwargs,
    )
    test_loader = DataLoader(
        WaveformDataset(load_metadata(csv_path, "test"), class_names, audio_cfg),
        **loader_kwargs,
    )
    return class_names, val_loader, test_loader


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device):
    model.eval()
    all_y, all_pred, all_logits, all_paths = [], [], [], []
    total_loss = 0.0
    criterion = nn.CrossEntropyLoss()

    for wav, y, paths, _, _ in loader:
        wav = wav.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(wav)
        loss = criterion(logits, y)
        total_loss += float(loss.item()) * y.size(0)
        pred = logits.argmax(dim=1)
        all_y.extend(y.cpu().tolist())
        all_pred.extend(pred.cpu().tolist())
        all_logits.append(logits.cpu().numpy())
        all_paths.extend(paths)

    return {
        "loss": total_loss / max(1, len(all_y)),
        "accuracy": accuracy_score(all_y, all_pred),
        "macro_f1": f1_score(
            all_y,
            all_pred,
            average="macro",
            zero_division=0,
        ),
        "y_true": np.asarray(all_y),
        "y_pred": np.asarray(all_pred),
        "logits": np.concatenate(all_logits, axis=0),
        "paths": all_paths,
    }


def save_eval(
    result: dict,
    output_dir: str | Path,
    split: str,
    class_names: Sequence[str],
) -> None:
    save_classification_artifacts(
        output_dir=output_dir,
        split=split,
        y_true=result["y_true"],
        scores=result["logits"],
        paths=result["paths"],
        class_names=class_names,
    )


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    class_names: Sequence[str],
    output_dir: str | Path,
    epochs: int = 25,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    patience: int = 6,
    device: str | None = None,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device_obj = torch.device(
        device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    model.to(device_obj)

    trainable = [p for p in model.parameters() if p.requires_grad]
    if not trainable:
        raise ValueError("No trainable parameters")
    optimizer = torch.optim.AdamW(
        trainable,
        lr=lr,
        weight_decay=weight_decay,
    )
    criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=device_obj.type == "cuda",
    )

    best_f1 = -1.0
    bad_epochs = 0
    history = []
    best_path = output_dir / "best.pt"

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        seen = 0
        for wav, y, _, _, _ in train_loader:
            wav = wav.to(device_obj, non_blocking=True)
            y = y.to(device_obj, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device_obj.type,
                enabled=device_obj.type == "cuda",
            ):
                logits = model(wav)
                loss = criterion(logits, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += float(loss.item()) * y.size(0)
            seen += y.size(0)

        val = evaluate(model, val_loader, device_obj)
        row = {
            "epoch": epoch,
            "train_loss": total_loss / max(1, seen),
            "val_loss": val["loss"],
            "val_accuracy": val["accuracy"],
            "val_macro_f1": val["macro_f1"],
        }
        history.append(row)
        print(json.dumps(row, ensure_ascii=False))

        if val["macro_f1"] > best_f1:
            best_f1 = val["macro_f1"]
            bad_epochs = 0
            torch.save(
                {
                    "model": model.state_dict(),
                    "class_names": list(class_names),
                },
                best_path,
            )
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

    pd.DataFrame(history).to_csv(
        output_dir / "history.csv",
        index=False,
    )
    checkpoint = torch.load(best_path, map_location=device_obj)
    model.load_state_dict(checkpoint["model"])

    val = evaluate(model, val_loader, device_obj)
    test = evaluate(model, test_loader, device_obj)
    save_eval(val, output_dir, "val", class_names)
    save_eval(test, output_dir, "test", class_names)
    return test


def add_common_args(
    parser: argparse.ArgumentParser,
    default_output: str | Path,
) -> None:
    parser.add_argument("--csv", default=str(config.METADATA_CSV))
    parser.add_argument("--classes", default=str(config.CLASS_NAMES_PATH))
    parser.add_argument("--output", default=str(default_output))
    parser.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    parser.add_argument("--epochs", type=int, default=config.EPOCHS)
    parser.add_argument("--lr", type=float, default=config.LEARNING_RATE)
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=config.WEIGHT_DECAY,
    )
    parser.add_argument("--patience", type=int, default=config.PATIENCE)
    parser.add_argument("--num-workers", type=int, default=config.NUM_WORKERS)
    parser.add_argument("--seed", type=int, default=config.SEED)
    parser.add_argument("--device", default=config.DEVICE)
    parser.add_argument(
        "--highpass-hz",
        type=float,
        default=config.HIGHPASS_HZ,
    )
    parser.add_argument(
        "--silence-threshold-dbfs",
        type=float,
        default=config.SILENCE_THRESHOLD_DBFS,
    )
    parser.add_argument(
        "--target-rms-dbfs",
        type=float,
        default=config.TARGET_RMS_DBFS,
    )


def audio_config_from_args(args) -> AudioConfig:
    return AudioConfig(
        highpass_hz=args.highpass_hz,
        silence_threshold_dbfs=args.silence_threshold_dbfs,
        target_rms_dbfs=args.target_rms_dbfs,
    )
