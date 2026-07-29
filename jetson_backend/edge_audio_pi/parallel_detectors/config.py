"""
Central configuration for the parallel specialized detection system.

Three independent detectors, each responsible for a different group of classes.
Each detector produces sigmoid probabilities per target class (no cross-detector
probability comparison). Per-class thresholds are tuned during evaluation.

Everything downstream (manifest builder, inference runner, evaluator) imports the
mappings from here so there is a single source of truth.
"""
from __future__ import annotations

import os
from pathlib import Path

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
PKG_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PKG_DIR.parent
EFFICIENTAT_DIR = PROJECT_ROOT / "EfficientAT"
DATASET_DIR = PROJECT_ROOT / "edge_audio_dataset"
MUSAN_DIR = DATASET_DIR / "musan_mixed_2s"
MUSAN_METADATA_CSV = DATASET_DIR / "musan_mix_metadata.csv"

MANIFEST_CSV = PKG_DIR / "manifest.csv"
SCORES_DIR = PKG_DIR / "scores"
CHECKPOINTS_DIR = PKG_DIR / "checkpoints"
REPORT_DIR = PKG_DIR / "report"

# AudioSet label maps (resolved by display_name, never by hard-coded index).
EFFICIENTAT_LABELS_CSV = EFFICIENTAT_DIR / "metadata" / "class_labels_indices.csv"

# --------------------------------------------------------------------------- #
# Audio properties (musan_mixed_2s clips: 16 kHz mono, exactly 2.0 s, float32)
# --------------------------------------------------------------------------- #
DATASET_SR = 16000
CLIP_SECONDS = 2.0
CLIP_SAMPLES = int(DATASET_SR * CLIP_SECONDS)  # 32000

EFFICIENTAT_SR = 32000  # mn10_as was trained at 32 kHz -> upsample 16k->32k
YAMNET_SR = 16000       # YAMNet expects 16 kHz mono float32 -> no resample needed

# --------------------------------------------------------------------------- #
# Class -> detector mapping
#
# Every folder in musan_mixed_2s/ maps to either:
#   (detector_name, canonical_class)   -> a target class for that detector
#   or None                            -> a negative / non-target class
#
# Korean folder names are kept as-is (dataset uses them). `canonical_class` is the
# internal English key used in score columns, metrics and reports.
# --------------------------------------------------------------------------- #
FOLDER_TO_TARGET: dict[str, tuple[str, str] | None] = {
    # --- YAMNet group (TF, zero-shot AudioSet) ---
    # 경보 and 화재경보 are MERGED into one `siren` class: the trained head could
    # not separate them (fire_alarm precision 0.49, mostly siren confusion) and the
    # downstream action for both is identical. Folder names stay distinct in the
    # score CSVs (`true_label`), so per-folder recall can still be inspected.
    "경보":            ("yamnet", "siren"),
    "화재경보":        ("yamnet", "siren"),
    "비명":            ("yamnet", "scream"),
    "울음":            ("yamnet", "crying"),
    # NOTE: plan listed "공사장 소리" (construction) for YAMNet but there is no such
    #       folder in the dataset, so it cannot be evaluated and is omitted.

    # --- EfficientAT group (mn10_as, zero-shot AudioSet) ---
    "자전거":          ("efficientat", "bicycle_bell"),
    "baby_cry":        ("efficientat", "baby_cry"),
    "총":              ("efficientat", "gunshot"),
    "유리깨지는소리":  ("efficientat", "glass_break"),

    # --- MobileNetV4-Small group (timm, trained) ---
    "물소리":          ("mobilenetv4", "water"),
    "knock":           ("mobilenetv4", "knock"),
    "dog_bark":        ("mobilenetv4", "dog"),
    "cat_meow":        ("mobilenetv4", "cat"),

    # --- Negatives (not assigned to any detector; FP sources) ---
    "background":      None,
    "car_horn":        None,
}

# Ordered target classes per detector (defines score-column order & report order).
DETECTOR_CLASSES: dict[str, list[str]] = {
    "yamnet":       ["siren", "scream", "crying"],
    "efficientat":  ["bicycle_bell", "baby_cry", "gunshot", "glass_break"],
    "mobilenetv4":  ["water", "knock", "dog", "cat"],
}

# canonical_class -> owning detector (derived; convenience lookup)
CLASS_TO_DETECTOR: dict[str, str] = {
    cls: det for det, classes in DETECTOR_CLASSES.items() for cls in classes
}

# --------------------------------------------------------------------------- #
# Zero-shot AudioSet mapping: canonical_class -> candidate AudioSet display_names.
# The detector resolves these to indices at runtime and takes the MAX probability
# across the candidates. Edit these lists to widen/narrow a detector's coverage.
# --------------------------------------------------------------------------- #
EFFICIENTAT_AUDIOSET_NAMES: dict[str, list[str]] = {
    "bicycle_bell": ["Bicycle bell"],
    "baby_cry":     ["Baby cry, infant cry"],
    "gunshot":      ["Gunshot, gunfire", "Machine gun", "Cap gun", "Fusillade"],
    "glass_break":  ["Shatter", "Glass", "Breaking"],
}

YAMNET_AUDIOSET_NAMES: dict[str, list[str]] = {
    # siren covers both 경보 and 화재경보 (merged class), so the fire-alarm AudioSet
    # names are folded into this candidate list.
    "siren":      ["Siren", "Civil defense siren", "Ambulance (siren)",
                   "Police car (siren)", "Fire engine, fire truck (siren)", "Alarm",
                   "Fire alarm", "Smoke detector, smoke alarm", "Buzzer"],
    "scream":     ["Screaming", "Shout", "Yell"],
    "crying":     ["Crying, sobbing", "Whimper"],
}

# --------------------------------------------------------------------------- #
# Split configuration (group-safe: no source group crosses train/val/test)
# --------------------------------------------------------------------------- #
GROUP_COLUMN = "group_id"          # from musan_mix_metadata.csv; falls back to source_file
SPLIT_FRACTIONS = {"train": 0.7, "val": 0.15, "test": 0.15}
SPLIT_SEED = 42

# --------------------------------------------------------------------------- #
# Model / preprocessing hyper-parameters
# --------------------------------------------------------------------------- #
EFFICIENTAT_MODEL_NAME = "mn10_as"
# mn10_as was trained on 10 s AudioSet clips and produces saturated/garbage logits
# on very short (2 s) inputs. Each clip is TILED (repeated) up to this length
# before the mel transform, which restores sane, well-calibrated predictions.
EFFICIENTAT_TARGET_SECONDS = 10.0

# YAMNet: how to aggregate the per-frame [n_frames, 521] scores into one score per
# clip. "max" is detection-oriented (event may occupy part of the clip); "mean"
# is smoother. Configurable here.
YAMNET_FRAME_AGG = "max"           # "max" | "mean"
YAMNET_HUB_HANDLE = "https://tfhub.dev/google/yamnet/1"

# MobileNetV4-Small (timm) trained detector
MOBILENETV4_TIMM_NAME = "mobilenetv4_conv_small.e2400_r224_in1k"
MOBILENETV4_IMG_SIZE = 224
MOBILENETV4_N_MELS = 128
MOBILENETV4_CKPT = CHECKPOINTS_DIR / "mobilenetv4_small.pt"

# Fine-tuned checkpoints for the (originally zero-shot) detectors. When these
# exist, run_inference uses the TRAINED head instead of zero-shot AudioSet mapping.
EFFICIENTAT_CKPT = CHECKPOINTS_DIR / "efficientat_head.pt"
YAMNET_CKPT = CHECKPOINTS_DIR / "yamnet_head.pt"


def get_device(prefer: str = "auto") -> str:
    """Resolve a torch device string: cuda -> mps -> cpu."""
    import torch
    if prefer != "auto":
        return prefer
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
        return "mps"
    return "cpu"
