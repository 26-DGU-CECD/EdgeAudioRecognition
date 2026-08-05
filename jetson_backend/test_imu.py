#!/usr/bin/env python3
"""IMU 단독 점검/캘리브레이션 도구.

  python3 test_imu.py --scan          연결된 I2C 주소 확인
  python3 test_imu.py --calibrate     평평한 곳에 두고 바이어스 저장
  python3 test_imu.py                 실시간 움직임 상태 출력
"""
from __future__ import annotations

import argparse
import sys
import time

from imu_module import (
    CALIBRATION_FILE,
    DEFAULT_ADDRESS,
    DEFAULT_BUS,
    DEFAULT_SAMPLE_HZ,
    KNOWN_CHIPS,
    REG_WHO_AM_I,
    ImuCalibration,
    ImuError,
    ImuModule,
)
from io_setup import configure_utf8_stdio
from motion_state import summarize_motion

STATE_COLORS = {
    "still": "\033[32m",
    "motion": "\033[33m",
    "shock": "\033[31m",
    "free_fall": "\033[31m",
    "unknown": "\033[90m",
}


def scan(bus_number: int) -> int:
    try:
        from smbus2 import SMBus
    except ImportError as exc:
        print(f"smbus2가 없습니다: {exc}", file=sys.stderr)
        return 1

    found = []
    with SMBus(bus_number) as bus:
        for address in range(0x03, 0x78):
            try:
                bus.read_byte(address)
            except OSError:
                continue
            found.append(address)

    if not found:
        print(f"i2c-{bus_number}에서 장치를 찾지 못했습니다.")
        return 1

    print(f"i2c-{bus_number} 응답 주소:")
    with SMBus(bus_number) as bus:
        for address in found:
            note = ""
            try:
                who_am_i = bus.read_byte_data(address, REG_WHO_AM_I)
                if who_am_i in KNOWN_CHIPS:
                    note = f" <- IMU 후보 {KNOWN_CHIPS[who_am_i]} (0x{who_am_i:02X})"
            except OSError:
                pass
            print(f"  0x{address:02X}{note}")
    return 0


def calibrate(module: ImuModule, seconds: float) -> int:
    print(
        f"기기를 평평한 곳에 놓고 움직이지 마세요. {seconds:.0f}초간 측정합니다.",
        flush=True,
    )
    for remaining in range(3, 0, -1):
        print(f"  {remaining}...", flush=True)
        time.sleep(1.0)

    calibration = module.calibrate(seconds)
    path = calibration.save(CALIBRATION_FILE)
    print("\n캘리브레이션 완료")
    print(f"  accel_bias  = {tuple(round(v, 5) for v in calibration.accel_bias)} g")
    print(f"  gyro_bias   = {tuple(round(v, 4) for v in calibration.gyro_bias)} dps")
    print(f"  accel_scale = {calibration.accel_scale:.5f}")
    print(f"  저장 위치    = {path}")

    time.sleep(0.5)
    snapshot = summarize_motion(module.recent_samples(1.0))
    print(f"\n검증(정지 상태 1초): {snapshot.summary()}")
    if snapshot.state != "still":
        print(
            "  경고: 정지로 판정되지 않았습니다. 진동 없는 곳에서 다시 시도하세요.",
            file=sys.stderr,
        )
    return 0


def monitor(module: ImuModule, window_seconds: float, duration: float) -> int:
    print(f"{module.description} | Ctrl+C로 종료", flush=True)
    print(
        f"{'time':>8} {'state':>10} {'a_rms(g)':>9} {'a_peak(g)':>10} "
        f"{'gyro(dps)':>10} {'pitch':>7} {'roll':>7} {'temp':>6}",
        flush=True,
    )

    started = time.monotonic()
    try:
        while duration <= 0.0 or time.monotonic() - started < duration:
            time.sleep(window_seconds)
            snapshot = summarize_motion(module.recent_samples(window_seconds))
            color = STATE_COLORS.get(snapshot.state, "")
            line = (
                f"{time.strftime('%H:%M:%S'):>8} {snapshot.state:>10} "
                f"{snapshot.accel_rms_g:>9.4f} {snapshot.accel_peak_g:>10.3f} "
                f"{snapshot.gyro_max_dps:>10.1f} {snapshot.pitch_deg:>7.1f} "
                f"{snapshot.roll_deg:>7.1f} {snapshot.temperature_c:>6.1f}"
            )
            print(f"{color}{line}\033[0m", flush=True)
    except KeyboardInterrupt:
        print("\n종료합니다.")

    if module.read_errors:
        print(f"I2C 읽기 오류 누계: {module.read_errors}", file=sys.stderr)
    return 0


def main(argv: list[str] | None = None) -> int:
    configure_utf8_stdio()
    parser = argparse.ArgumentParser(description="MPU9250 IMU 점검 도구")
    parser.add_argument("--bus", type=int, default=DEFAULT_BUS)
    parser.add_argument(
        "--address",
        type=lambda value: int(value, 0),
        default=DEFAULT_ADDRESS,
    )
    parser.add_argument("--sample-hz", type=float, default=DEFAULT_SAMPLE_HZ)
    parser.add_argument(
        "--window-seconds",
        type=float,
        default=1.0,
        help="한 줄에 요약할 구간 길이. 오디오 hop과 맞추면 비교하기 좋습니다.",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=0.0,
        help="0이면 Ctrl+C까지 계속합니다.",
    )
    parser.add_argument("--scan", action="store_true", help="I2C 주소만 훑고 끝냅니다.")
    parser.add_argument("--calibrate", action="store_true")
    parser.add_argument("--calibrate-seconds", type=float, default=3.0)
    parser.add_argument(
        "--raw",
        action="store_true",
        help="저장된 캘리브레이션을 무시하고 원시값을 씁니다.",
    )
    args = parser.parse_args(argv)

    if args.scan:
        return scan(args.bus)

    calibration = None if args.raw else ImuCalibration.load(CALIBRATION_FILE)
    try:
        module = ImuModule(
            bus=args.bus,
            address=args.address,
            sample_hz=args.sample_hz,
            history_seconds=max(4.0, args.window_seconds * 2.0),
            calibration=calibration,
        )
    except ImuError as exc:
        print(f"IMU 초기화 실패: {exc}", file=sys.stderr)
        return 1

    with module:
        if args.calibrate:
            try:
                return calibrate(module, args.calibrate_seconds)
            except ImuError as exc:
                print(f"캘리브레이션 실패: {exc}", file=sys.stderr)
                return 1
        time.sleep(min(1.0, args.window_seconds))
        return monitor(module, args.window_seconds, args.duration)


if __name__ == "__main__":
    raise SystemExit(main())
