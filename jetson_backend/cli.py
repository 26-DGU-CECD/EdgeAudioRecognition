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
        "--db-offset",
        type=float,
        default=runtime_config.DEFAULT_DB_OFFSET,
        help=(
            "dBFS를 앱이 쓰는 양수 dB로 변환합니다: db=max(0, dBFS+offset). "
            "기본값 90은 --min-db 45(= -45 dBFS)와 앱의 db>=45 조건을 일치시킵니다."
        ),
    )
    parser.add_argument(
        "--full-packet",
        action="store_true",
        help="items/dbfs 등 부가 필드를 포함합니다. JSON이 커지니 MTU를 확인하세요.",
    )
    parser.add_argument(
        "--north-offset",
        type=float,
        default=runtime_config.DEFAULT_NORTH_OFFSET,
        help=(
            "angle = (DOAANGLE 원값 - offset) %% 360. "
            "보정 후 angle 0이 착용자 뒤쪽, 180이 앞쪽이 되도록 잡습니다."
        ),
    )
    parser.add_argument(
        "--disable-doa",
        action="store_true",
        help="ReSpeaker USB DOA 읽기를 끕니다 (angle은 null로 전송).",
    )
    parser.add_argument(
        "--doa-poll-interval",
        type=float,
        default=runtime_config.DEFAULT_DOA_POLL_INTERVAL,
        help="USB DSP DOA 폴링 주기(초). 기본 0.1.",
    )
    parser.add_argument(
        "--disable-imu",
        action="store_true",
        help="MPU-9250 흔들림 보정을 끕니다 (DOA 원값을 그대로 사용).",
    )
    parser.add_argument(
        "--imu-bus",
        type=int,
        default=runtime_config.DEFAULT_IMU_BUS,
        help="MPU-9250이 붙은 I2C 버스 번호입니다.",
    )
    parser.add_argument(
        "--imu-address",
        type=lambda value: int(value, 0),
        default=runtime_config.DEFAULT_IMU_ADDRESS,
        help="MPU-9250 I2C 주소 (기본 0x68).",
    )
    parser.add_argument(
        "--imu-poll-hz",
        type=float,
        default=runtime_config.DEFAULT_IMU_POLL_HZ,
        help="자이로 적분 주기(Hz). 기본 100.",
    )
    parser.add_argument(
        "--imu-turn-threshold",
        type=float,
        default=runtime_config.DEFAULT_IMU_TURN_THRESHOLD,
        help=(
            "yaw 각속도의 이동 평균이 이 값(deg/s)을 넘고, 그 창 안에서 회전 방향이 "
            "한 번도 뒤집히지 않았을 때만 '방향 전환'으로 봅니다. 흔들림은 반주기마다 "
            "반드시 뒤집히므로 진폭이 아무리 커도 걸리지 않습니다."
        ),
    )
    parser.add_argument(
        "--imu-turn-window",
        type=float,
        default=runtime_config.DEFAULT_IMU_TURN_WINDOW_SECONDS,
        help="방향 전환 판정 창(초). 흔들림 반주기보다 길어야 합니다. 기본 1.2.",
    )
    parser.add_argument(
        "--imu-ref-tau-still",
        type=float,
        default=runtime_config.DEFAULT_IMU_REF_TAU_STILL,
        help="정지/흔들림 상태에서 기준 방위가 따라가는 시정수(초). 클수록 흔들림을 잘 지웁니다.",
    )
    parser.add_argument(
        "--imu-ref-tau-turn",
        type=float,
        default=runtime_config.DEFAULT_IMU_REF_TAU_TURN,
        help="방향 전환 중 기준 방위가 따라가는 시정수(초). 작을수록 전환을 빨리 반영합니다.",
    )
    parser.add_argument(
        "--imu-calibration-seconds",
        type=float,
        default=runtime_config.DEFAULT_IMU_CALIBRATION_SECONDS,
        help="시작 시 자이로 바이어스를 평균낼 시간(초). 이 동안 정지해 있어야 합니다.",
    )
    parser.add_argument(
        "--imu-gyro-range",
        type=int,
        choices=(250, 500, 1000, 2000),
        default=runtime_config.DEFAULT_IMU_GYRO_RANGE_DPS,
        help="자이로 측정 범위(deg/s). 흔들림이 세면 클리핑을 피하려 크게 잡습니다.",
    )
    parser.add_argument(
        "--imu-yaw-axis",
        choices=("gravity", "z"),
        default=runtime_config.DEFAULT_IMU_YAW_AXIS,
        help=(
            "yaw 회전축. gravity는 가속도로 측정한 중력축에 자이로를 투영하고, "
            "z는 칩의 자이로 Z를 그대로 씁니다. 보드가 수평이면 둘은 같습니다."
        ),
    )
    parser.add_argument(
        "--imu-swing-sign",
        type=float,
        choices=(1.0, -1.0),
        default=runtime_config.DEFAULT_IMU_SWING_SIGN,
        help=(
            "angle = DOA + sign*swing - north_offset. 자이로와 DOA의 회전 방향이 "
            "반대면 -1로 뒤집습니다 (흔들림이 2배로 커지면 부호가 틀린 것입니다)."
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
