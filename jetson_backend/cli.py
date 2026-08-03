"""Command-line interface for Raspberry Pi parallel inference."""
from __future__ import annotations

import argparse

import runtime_config
from constants import MIC_CHANNEL_INDEX


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "ReSpeaker 실시간 병렬 전문 검출 "
            "(EfficientAT + YAMNet + MobileNetV4)"
        )
    )
    parser.add_argument("--list-devices", action="store_true")
    parser.add_argument("--device-index", type=int, default=None)
    parser.add_argument("--channel-index", type=int, default=MIC_CHANNEL_INDEX)
    parser.add_argument(
        "--input-wav",
        default=None,
        help="마이크 대신 WAV 파일로 전체 추론 경로를 확인합니다.",
    )
    parser.add_argument(
        "--hop-seconds",
        type=float,
        default=runtime_config.DEFAULT_HOP_SECONDS,
    )
    parser.add_argument(
        "--detectors",
        default=",".join(runtime_config.DEFAULT_DETECTORS),
        help="efficientat,yamnet,mobilenetv4 중 실행할 항목",
    )
    parser.add_argument("--thresholds", default=None)
    parser.add_argument("--checkpoints-dir", default=None)
    parser.add_argument("--efficientat-dir", default=None)
    parser.add_argument(
        "--min-db",
        type=float,
        default=runtime_config.DEFAULT_MIN_DB,
        help="45는 -45 dBFS를 뜻합니다.",
    )
    parser.add_argument(
        "--no-skip-low-db",
        dest="skip_low_db",
        action="store_false",
        help="저음량 구간도 추론합니다.",
    )
    parser.add_argument(
        "--skip-low-db",
        dest="skip_low_db",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.set_defaults(skip_low_db=True)
    parser.add_argument(
        "--debounce-seconds",
        type=float,
        default=runtime_config.DEFAULT_DEBOUNCE_SECONDS,
    )
    parser.add_argument("--concurrent", action="store_true")
    parser.add_argument(
        "--torch-threads",
        type=int,
        default=runtime_config.DEFAULT_TORCH_THREADS,
    )
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--no-ble", action="store_true")
    parser.add_argument("--ble-name", default="JHello")
    parser.add_argument(
        "--ble-chunk-bytes",
        type=int,
        default=244,
        help=(
            "경고 임계값입니다. 패킷은 항상 단일 프레임으로 보내며, "
            "이 값을 넘으면 경고만 출력합니다."
        ),
    )
    parser.add_argument(
        "--no-battery",
        dest="battery",
        action="store_false",
        help="UPS 배터리 잔량을 전송하지 않습니다.",
    )
    parser.set_defaults(battery=True)
    parser.add_argument(
        "--battery-bus",
        type=int,
        default=runtime_config.DEFAULT_BATTERY_BUS,
        help="UPS 연료계가 붙은 I2C 버스 번호입니다.",
    )
    parser.add_argument(
        "--battery-interval",
        type=float,
        default=runtime_config.DEFAULT_BATTERY_INTERVAL_SECONDS,
        help="배터리를 다시 읽는 최소 간격(초)입니다.",
    )

    args = parser.parse_args(argv)
    args.detector_names = [
        name.strip()
        for name in args.detectors.split(",")
        if name.strip()
    ]
    unknown = [
        name
        for name in args.detector_names
        if name not in runtime_config.DEFAULT_DETECTORS
    ]
    if not args.detector_names:
        parser.error("--detectors에 최소 한 개를 지정하세요.")
    if unknown:
        parser.error(f"알 수 없는 detector: {unknown}")
    return args
