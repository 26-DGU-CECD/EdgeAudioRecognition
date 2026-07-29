"""Validate and load the three trained specialized detector checkpoints."""
from __future__ import annotations

import sys
import time
from typing import Dict, List

import runtime_config


def _require_checkpoint(name: str):
    checkpoint = runtime_config.CHECKPOINTS[name]
    if not checkpoint.is_file():
        raise RuntimeError(
            f"{name} 체크포인트가 없습니다: {checkpoint}\n"
            "--checkpoints-dir 또는 EDGEAUDIO_CHECKPOINTS_DIR를 확인하세요."
        )
    return checkpoint


def configure_torch_threads(threads: int | None = None) -> int:
    import torch

    count = int(threads or runtime_config.DEFAULT_TORCH_THREADS)
    if count <= 0:
        raise ValueError("torch threads는 1 이상이어야 합니다.")
    torch.set_num_threads(count)
    return count


def build_detector(name: str, batch_size: int = 1):
    checkpoint = _require_checkpoint(name)

    if name == "efficientat":
        if not runtime_config.EFFICIENTAT_DIR.is_dir():
            raise RuntimeError(
                f"EfficientAT 최소 소스가 없습니다: {runtime_config.EFFICIENTAT_DIR}"
            )
        from edge_audio_pi.parallel_detectors.detectors.efficientat_detector import (
            EfficientATDetector,
        )

        return EfficientATDetector(batch_size=batch_size, ckpt=checkpoint)

    if name == "yamnet":
        from edge_audio_pi.parallel_detectors.detectors.yamnet_detector import (
            YAMNetDetector,
        )

        return YAMNetDetector(ckpt=checkpoint)

    if name == "mobilenetv4":
        from edge_audio_pi.parallel_detectors.detectors.mobilenetv4_detector import (
            MobileNetV4Detector,
        )

        return MobileNetV4Detector(ckpt_path=checkpoint, batch_size=batch_size)

    raise ValueError(f"알 수 없는 detector: {name}")


def build_detectors(
    names: List[str] | None = None,
    *,
    batch_size: int = 1,
    torch_threads: int | None = None,
    verbose: bool = True,
) -> Dict[str, object]:
    requested = list(names or runtime_config.DEFAULT_DETECTORS)
    unknown = [name for name in requested if name not in runtime_config.CHECKPOINTS]
    if unknown:
        raise ValueError(f"알 수 없는 detector: {unknown}")

    configured_threads = configure_torch_threads(torch_threads)
    if verbose:
        print(f"torch threads={configured_threads}", flush=True)

    detectors: Dict[str, object] = {}
    for name in requested:
        started = time.perf_counter()
        detectors[name] = build_detector(name, batch_size=batch_size)
        if verbose:
            print(
                f"[load] {name}: {time.perf_counter() - started:.1f}s",
                file=sys.stderr,
                flush=True,
            )
    return detectors
