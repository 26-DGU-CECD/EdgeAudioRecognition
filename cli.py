from __future__ import annotations

import argparse
from pathlib import Path

from realtime_inference import (
    DEFAULT_ENHANCE_SHARPNESS,
    DEFAULT_ENHANCE_THRESHOLD_DB,
    DEFAULT_MAIN_GAIN_DB,
    DEFAULT_MIN_DB,
    DEFAULT_MIN_SCORE,
    DEFAULT_NOISE_REDUCTION_DB,
    MIC_CHANNEL_INDEX,
    db_gate_threshold,
)


def parse_optional_db_gate(value: str) -> float | None:
    normalized = str(value).strip().lower()
    if normalized in {"off", "none", "disable", "disabled", "all"}:
        return None
    try:
        return float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "expected a number, or one of: off, none, disabled, all"
        ) from exc


def format_optional_db_gate(value: float | None) -> str:
    if value is None:
        return "off"
    return f"{db_gate_threshold(value):+.1f}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Realtime EfficientAT inference from ReSpeaker Array V3 over BLE, "
            "with ReSpeaker DOA fields for EdgeAudioRecognition."
        )
    )
    parser.add_argument(
        "--efficientat-dir",
        default=str(Path(__file__).resolve().parent / "EfficientAT"),
        help="Path to cloned fschmid56/EfficientAT repository.",
    )
    parser.add_argument("--device-index", type=int, default=None)
    parser.add_argument("--channel-index", type=int, default=MIC_CHANNEL_INDEX)
    parser.add_argument("--list-devices", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--min-db", type=float, default=DEFAULT_MIN_DB)
    parser.add_argument("--enhance-threshold-db", type=float, default=DEFAULT_ENHANCE_THRESHOLD_DB)
    parser.add_argument("--noise-reduction-db", type=float, default=DEFAULT_NOISE_REDUCTION_DB)
    parser.add_argument("--main-gain-db", type=float, default=DEFAULT_MAIN_GAIN_DB)
    parser.add_argument("--gain-db", type=float, dest="main_gain_db", help=argparse.SUPPRESS)
    parser.add_argument("--enhance-sharpness", type=float, default=DEFAULT_ENHANCE_SHARPNESS)
    parser.add_argument("--min-score", type=float, default=DEFAULT_MIN_SCORE)
    parser.add_argument("--ble-name", default="JHello")
    parser.add_argument("--ble-chunk-bytes", type=int, default=244)
    parser.add_argument("--north-offset", type=float, default=0.0)
    parser.add_argument("--disable-doa", action="store_true")
    parser.add_argument("--doa-source", choices=("auto", "audio", "usb"), default="auto")
    parser.add_argument("--doa-poll-interval", type=float, default=0.1)
    parser.add_argument("--audio-doa-min-db", type=parse_optional_db_gate, default=None)
    parser.add_argument("--audio-doa-window-ms", type=float, default=250.0)
    parser.add_argument("--db-offset", type=float, default=80.0)
    parser.add_argument("--full-packet", action="store_true")
    parser.add_argument(
        "--skip-inference-below-threshold",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip model inference when chunk dBFS is below --min-db. Default: true.",
    )
    return parser.parse_args()
