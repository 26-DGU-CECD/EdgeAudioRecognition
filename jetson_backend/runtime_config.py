"""Runtime paths shared with the trained detector evaluation configuration."""
from __future__ import annotations

import os
from pathlib import Path

from edge_audio_pi.parallel_detectors import config as detector_config

BACKEND_DIR = Path(__file__).resolve().parent
SAMPLE_RATE = detector_config.DATASET_SR
WINDOW_SECONDS = detector_config.CLIP_SECONDS
WINDOW_SAMPLES = detector_config.CLIP_SAMPLES
DETECTOR_CLASSES = detector_config.DETECTOR_CLASSES
CLASS_TO_DETECTOR = detector_config.CLASS_TO_DETECTOR
ALL_CLASSES = [
    class_name
    for detector_name in ("efficientat", "yamnet", "mobilenetv4")
    for class_name in DETECTOR_CLASSES[detector_name]
]

CHECKPOINT_FILES = {
    "efficientat": detector_config.EFFICIENTAT_CKPT.name,
    "yamnet": detector_config.YAMNET_CKPT.name,
    "mobilenetv4": detector_config.MOBILENETV4_CKPT.name,
}


def _env_path(name: str) -> Path | None:
    value = os.environ.get(name)
    return Path(value).expanduser().resolve() if value else None


CHECKPOINTS_DIR = (
    _env_path("EDGEAUDIO_CHECKPOINTS_DIR") or detector_config.CHECKPOINTS_DIR
)
THRESHOLDS_JSON = (
    _env_path("EDGEAUDIO_THRESHOLDS")
    or detector_config.REPORT_DIR / "thresholds.json"
)
EFFICIENTAT_DIR = (
    _env_path("EDGEAUDIO_EFFICIENTAT_DIR") or detector_config.EFFICIENTAT_DIR
)
CHECKPOINTS = {
    name: CHECKPOINTS_DIR / filename
    for name, filename in CHECKPOINT_FILES.items()
}

DEFAULT_DETECTORS = ["efficientat", "yamnet", "mobilenetv4"]
DEFAULT_HOP_SECONDS = 1.0
DEFAULT_MIN_DB = 45.0
DEFAULT_DEBOUNCE_SECONDS = 3.0
DEFAULT_TORCH_THREADS = min(4, os.cpu_count() or 4)
QUEUE_MAX_SECONDS = 6.0
MAX_WINDOWS_PER_CYCLE = 2


def apply_path_overrides(
    *,
    checkpoints_dir: str | Path | None = None,
    thresholds: str | Path | None = None,
    efficientat_dir: str | Path | None = None,
) -> None:
    global CHECKPOINTS_DIR, THRESHOLDS_JSON, EFFICIENTAT_DIR, CHECKPOINTS

    if checkpoints_dir:
        CHECKPOINTS_DIR = Path(checkpoints_dir).expanduser().resolve()
        CHECKPOINTS = {
            name: CHECKPOINTS_DIR / filename
            for name, filename in CHECKPOINT_FILES.items()
        }
        detector_config.CHECKPOINTS_DIR = CHECKPOINTS_DIR
        detector_config.EFFICIENTAT_CKPT = CHECKPOINTS["efficientat"]
        detector_config.YAMNET_CKPT = CHECKPOINTS["yamnet"]
        detector_config.MOBILENETV4_CKPT = CHECKPOINTS["mobilenetv4"]

    if thresholds:
        THRESHOLDS_JSON = Path(thresholds).expanduser().resolve()

    if efficientat_dir:
        EFFICIENTAT_DIR = Path(efficientat_dir).expanduser().resolve()
        detector_config.EFFICIENTAT_DIR = EFFICIENTAT_DIR
        detector_config.EFFICIENTAT_LABELS_CSV = (
            EFFICIENTAT_DIR / "metadata" / "class_labels_indices.csv"
        )


apply_path_overrides(
    checkpoints_dir=_env_path("EDGEAUDIO_CHECKPOINTS_DIR"),
    thresholds=_env_path("EDGEAUDIO_THRESHOLDS"),
    efficientat_dir=_env_path("EDGEAUDIO_EFFICIENTAT_DIR"),
)
