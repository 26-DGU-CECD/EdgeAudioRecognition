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
# 앱은 db >= 45.0 일 때만 홈 화면에 표시한다(sound_packet.dart:isDisplayable).
# db = dBFS + DEFAULT_DB_OFFSET 이므로 offset 90은 그 조건을 -45 dBFS와 같게 만들어
# DEFAULT_MIN_DB 게이트와 정확히 일치시킨다. (DB_OFFSET = MIN_DB + 45)
DEFAULT_DB_OFFSET = 90.0
DEFAULT_NORTH_OFFSET = 0.0
DEFAULT_DOA_POLL_INTERVAL = 0.1
DEFAULT_DEBOUNCE_SECONDS = 3.0
DEFAULT_TORCH_THREADS = min(4, os.cpu_count() or 4)
DEFAULT_BATTERY_BUS = 1
DEFAULT_BATTERY_INTERVAL_SECONDS = 5.0

# MPU-9250 yaw de-swing (imu.py). Bus 1 / 0x68 is where the module sits on this
# unit, sharing the bus with the 0x36 UPS fuel gauge.
DEFAULT_IMU_BUS = 1
DEFAULT_IMU_ADDRESS = 0x68
DEFAULT_IMU_POLL_HZ = 100.0
# deg/s. Magnitude half of the shake/turn gate; the reversal-count half lives in
# imu.py and is what actually rejects swings, so this only has to clear sensor
# noise and residual bias (~0.3 deg/s measured). It is deliberately well below
# the 30 deg/s a 90-degree turn taking 3 s produces: at 30 that ordinary turn
# sat exactly on the boundary, missed the gate, and took 13 s to track instead
# of 1.8 s.
DEFAULT_IMU_TURN_THRESHOLD = 10.0
DEFAULT_IMU_TURN_WINDOW_SECONDS = 1.2
# Reference-heading time constants. "still" is ~10x slower than the slowest
# swing period, so a shake passes through to `swing` nearly untouched, but not
# so slow that the few degrees left over after turning *while* swinging sit
# there for a minute; measured, 4 s trades 1.8->2.5 deg of shake leakage for
# 13.5->6 deg less post-turn residual. "turn" is short enough that a detected
# turn is absorbed within about a second, so direction changes survive.
DEFAULT_IMU_REF_TAU_STILL = 4.0
DEFAULT_IMU_REF_TAU_TURN = 0.25
DEFAULT_IMU_CALIBRATION_SECONDS = 1.5
# +-500 dps: a brisk keyring shake clips the +-250 dps range, and a clipped
# sample corrupts the yaw integral permanently.
DEFAULT_IMU_GYRO_RANGE_DPS = 500
DEFAULT_IMU_YAW_AXIS = "gravity"
# Gyro Z and the DSP's DOA may count in opposite rotational senses. Measured on
# this unit; if de-swing ever doubles the shake instead of cancelling it, this
# is the sign to flip.
DEFAULT_IMU_SWING_SIGN = 1.0
# Largest DOA/IMU timestamp mismatch still worth correcting, in seconds. Beyond
# it the swing describes a different moment than the angle and would add error.
DEFAULT_IMU_MAX_SYNC_AGE = 0.25
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
