from __future__ import annotations

import argparse
from pathlib import Path

import torch

import config
from common_torch import (
    audio_config_from_args,
    evaluate,
    make_eval_loaders,
    save_eval,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        choices=["efficientat", "mobilenetv4"],
        required=True,
    )
    parser.add_argument("--csv", default=str(config.METADATA_CSV))
    parser.add_argument("--classes", default=str(config.CLASS_NAMES_PATH))
    parser.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    parser.add_argument("--num-workers", type=int, default=config.NUM_WORKERS)
    parser.add_argument("--device", default=config.DEVICE)
    parser.add_argument("--highpass-hz", type=float, default=config.HIGHPASS_HZ)
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
    args = parser.parse_args()

    if args.model == "efficientat":
        from train_efficientat import EfficientATBaseline
        output = Path(config.EFFICIENTAT_OUTPUT)
        model = EfficientATBaseline(
            config.EFFICIENTAT_ROOT,
            config.EFFICIENTAT_MODEL_NAME,
            14,
            freeze_backbone=True,
        )
    else:
        from train_mobilenetv4 import MobileNetV4Baseline
        output = Path(config.MOBILENETV4_OUTPUT)
        model = MobileNetV4Baseline(14, freeze_backbone=True)

    class_names, val_loader, test_loader = make_eval_loaders(
        args.csv,
        args.classes,
        audio_config_from_args(args),
        args.batch_size,
        args.num_workers,
    )
    checkpoint_path = output / "best.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if list(checkpoint.get("class_names", class_names)) != list(class_names):
        raise ValueError("Checkpoint class order differs from class_names.txt")
    model.load_state_dict(checkpoint["model"])

    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    model.to(device)
    val = evaluate(model, val_loader, device)
    test = evaluate(model, test_loader, device)
    save_eval(val, output, "val", class_names)
    save_eval(test, output, "test", class_names)


if __name__ == "__main__":
    main()
