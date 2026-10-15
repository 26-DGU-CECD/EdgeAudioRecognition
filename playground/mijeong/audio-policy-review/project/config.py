"""Hardcoded project paths and D/E/F/G experiment settings."""
from __future__ import annotations

from pathlib import Path


# =============================================================================
# 1. EDIT THESE TWO PATHS
# =============================================================================
PROJECT_ROOT = Path("/home/plass/YAMNet/audio_experiments_DEF")
DATASET_ROOT = Path("/home/plass/YAMNet/aug_data")

EFFICIENTAT_ROOT = PROJECT_ROOT / "third_party" / "EfficientAT"

# =============================================================================
# 2. COMMON FILES
# =============================================================================
CLASS_NAMES_PATH = PROJECT_ROOT / "class_names.txt"
METADATA_CSV = PROJECT_ROOT / "metadata.csv"
DATASET_AUDIT_CSV = PROJECT_ROOT / "dataset_audit.csv"

EFFICIENTAT_OUTPUT = PROJECT_ROOT / "runs" / "A_efficientat"
YAMNET_OUTPUT = PROJECT_ROOT / "runs" / "B_yamnet"
MOBILENETV4_OUTPUT = PROJECT_ROOT / "runs" / "C_mobilenetv4"
FULL_ENSEMBLE_OUTPUT = PROJECT_ROOT / "runs" / "D_full_ensemble"

# E/F: predefined specialist assignment.
PREDEFINED_SPECIALIST_METADATA_ROOT = (
    PROJECT_ROOT / "specialist_metadata" / "predefined"
)
PREDEFINED_SPECIALIST_RUN_ROOT = (
    PROJECT_ROOT / "runs" / "E_predefined_specialists"
)
PREDEFINED_ENSEMBLE_ROOT = (
    PROJECT_ROOT / "runs" / "F_predefined_weighted_ensemble"
)

# G: automatic specialist assignment and its weighted ensemble.
AUTO_SPECIALIST_METADATA_ROOT = (
    PROJECT_ROOT / "specialist_metadata" / "automatic"
)
AUTO_SPECIALIST_RUN_ROOT = (
    PROJECT_ROOT / "runs" / "G_auto_specialists"
)
AUTO_ENSEMBLE_ROOT = (
    PROJECT_ROOT / "runs" / "G_auto_weighted_ensemble"
)

SPECIALIST_SWEEP_SUMMARY = (
    PROJECT_ROOT / "runs" / "specialist_ratio_sweep"
)

# =============================================================================
# 3. AUDIO / SPLIT
# =============================================================================
SEED = 42
TRAIN_RATIO = 0.80
VAL_RATIO = 0.10
TEST_RATIO = 0.10
GROUP_REGEX = None

SAMPLE_RATE = 16_000
CLIP_SECONDS = 2.0
HIGHPASS_HZ = 80.0
SILENCE_THRESHOLD_DBFS = -30.0
TARGET_RMS_DBFS = -20.0

# =============================================================================
# 4. TRAINING
# =============================================================================
BATCH_SIZE = 40
EPOCHS = 25
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
PATIENCE = 6
NUM_WORKERS = 4
DEVICE = None

EFFICIENTAT_MODEL_NAME = "mn04_as"

MODEL_NAMES = ("efficientat", "yamnet", "mobilenetv4")

# =============================================================================
# 5. E: PREDEFINED SPECIALIST CLASSES
# =============================================================================
# background를 제외한 13개 클래스가 정확히 한 모델에만 들어가야 합니다.
SPECIALIST_EXPERT_CLASSES = {
    "efficientat": [
        # 사전에 정한 클래스
        "bicycle",
        "baby_cry",
        "gun",
        "glass_breaking",
    ],
    "yamnet": [
        # 사전에 정한 클래스
        "siren",
        "fire_alarm",
        "scream",
        "cry",
        "car_horn",
    ],
    "mobilenetv4": [
        # 사전에 정한 클래스
        "water",
        "knock",
        "dog_bark",
        "cat_meow",
    ],
}

# 선택 사항. 전문 클래스와 중복되면 안 됩니다.
SPECIALIST_HARD_NEGATIVE_CLASSES = {
    "efficientat": [],
    "yamnet": [],
    "mobilenetv4": [],
}

# =============================================================================
# 6. E/G: NON-EXPERT DATA RATIO SWEEP
# =============================================================================
SPECIALIST_NON_EXPERT_FRACTIONS = [
    0.70,
    0.50,
    0.30,
    0.15,
]

# hard-negative는 최소 이 비율을 유지합니다.
SPECIALIST_HARD_NEGATIVE_MIN_FRACTION = 0.50

# =============================================================================
# 7. G: AUTOMATIC SPECIALIST ASSIGNMENT
# =============================================================================
# 각 foreground 클래스는 A/B/C baseline validation per-class F1이 가장 높은
# 모델에 자동 배정됩니다.
AUTO_MIN_EXPERT_CLASSES_PER_MODEL = 1

# 동률 또는 매우 근접한 경우에도 argmax로 한 모델에만 배정합니다.
# 각 모델에 최소 전문 클래스 수를 만족시키기 위해 필요한 경우,
# validation F1 손실(regret)이 가장 작은 클래스를 재배정합니다.

# 자동 hard-negative:
# 해당 모델에서 실제 비전문 클래스가 그 모델의 전문 클래스로 잘못 예측된
# 비율이 높은 순서로 선택합니다.
AUTO_HARD_NEGATIVE_CLASS_COUNT = 3

# =============================================================================
# 8. F/G: WEIGHTED ENSEMBLE
# =============================================================================
SPECIALIST_WEIGHT_FLOOR = 0.05
SPECIALIST_WEIGHT_POWER = 2.0
SPECIALIST_EXPERT_WEIGHT_BOOST = 1.25


def ratio_tag(fraction: float) -> str:
    value = float(fraction)
    if not 0.0 < value <= 1.0:
        raise ValueError(f"Ratio must be in (0, 1], got {fraction}")
    return f"nonexpert_{int(round(value * 100)):03d}"


def predefined_metadata_dir(fraction: float) -> Path:
    return PREDEFINED_SPECIALIST_METADATA_ROOT / ratio_tag(fraction)


def predefined_csv(model_name: str, fraction: float) -> Path:
    if model_name not in MODEL_NAMES:
        raise ValueError(f"Unknown model name: {model_name}")
    return predefined_metadata_dir(fraction) / f"{model_name}.csv"


def predefined_output(model_name: str, fraction: float) -> Path:
    if model_name not in MODEL_NAMES:
        raise ValueError(f"Unknown model name: {model_name}")
    return PREDEFINED_SPECIALIST_RUN_ROOT / ratio_tag(fraction) / model_name


def predefined_ensemble_output(fraction: float) -> Path:
    return PREDEFINED_ENSEMBLE_ROOT / ratio_tag(fraction)


def auto_metadata_dir(fraction: float) -> Path:
    return AUTO_SPECIALIST_METADATA_ROOT / ratio_tag(fraction)


def auto_csv(model_name: str, fraction: float) -> Path:
    if model_name not in MODEL_NAMES:
        raise ValueError(f"Unknown model name: {model_name}")
    return auto_metadata_dir(fraction) / f"{model_name}.csv"


def auto_output(model_name: str, fraction: float) -> Path:
    if model_name not in MODEL_NAMES:
        raise ValueError(f"Unknown model name: {model_name}")
    return AUTO_SPECIALIST_RUN_ROOT / ratio_tag(fraction) / model_name


def auto_ensemble_output(fraction: float) -> Path:
    return AUTO_ENSEMBLE_ROOT / ratio_tag(fraction)


def validate_paths(
    require_dataset: bool = False,
    require_efficientat: bool = False,
) -> None:
    if not PROJECT_ROOT.exists():
        raise FileNotFoundError(
            f"PROJECT_ROOT does not exist: {PROJECT_ROOT}\n"
            "Edit PROJECT_ROOT in config.py."
        )
    if require_dataset and not DATASET_ROOT.exists():
        raise FileNotFoundError(
            f"DATASET_ROOT does not exist: {DATASET_ROOT}\n"
            "Edit DATASET_ROOT in config.py."
        )
    if not CLASS_NAMES_PATH.exists():
        raise FileNotFoundError(f"class_names.txt not found: {CLASS_NAMES_PATH}")
    if require_efficientat and not EFFICIENTAT_ROOT.exists():
        raise FileNotFoundError(
            f"EfficientAT repository not found: {EFFICIENTAT_ROOT}"
        )
