#!/usr/bin/env python3
"""Measure which way `--imu-swing-sign` has to point on this unit.

`angle = raw_DOA + sign*swing - north_offset` only cancels the keyring's swing
if the gyro and the DSP count rotation in the same direction. They may not:
`usb_4_mic_array/tuning.py` documents DOAANGLE as "Orientation depends on build
configuration", so the sense is a property of the firmware image on this device
and cannot be looked up. It has to be measured.

Getting it backwards is the worst possible outcome — the correction then adds
the swing instead of removing it and the displayed direction wobbles *twice* as
much as with no IMU at all. Eyeballing that on the live display works but is
crude, so this fits a regression instead and reports how well it fit.

Physically, with a fixed source: raw_DOA = source - k*yaw, where k is +1 when
the two agree and -1 when they oppose. So the fitted slope d(raw_DOA)/d(yaw) is
-k, and the sign to configure is k = -slope.

Usage
-----
Put a *continuous* sound near the unit (music, a running tap, a fan — not
speech, which gaps), keep the source and yourself still, then rotate only the
keyring smoothly back and forth by roughly +-60 degrees for the whole run:

    ./venv/bin/python imu_sign_check.py --seconds 25
"""
from __future__ import annotations

import argparse
import sys
import time

import runtime_config
from doa import DOAReader
from imu import IMUReader, wrap180

# Below this the rotation was too small to tell the two conventions apart, and
# a slope fitted through noise would be worse than no answer.
MIN_YAW_SPAN_DEG = 40.0
# The fit should land near +-1. Far from it means the DSP angle was not tracking
# the rotation (moving source, reflections, too little sound) rather than that
# some other scale factor is correct.
SLOPE_MIN, SLOPE_MAX = 0.4, 2.5
MIN_R_SQUARED = 0.5


def _unwrapped(previous: float | None, total: float, current: float) -> float:
    """Accumulate a wrapping angle into a continuous one."""
    if previous is None:
        return 0.0
    return total + wrap180(current - previous)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--seconds", type=float, default=25.0)
    parser.add_argument("--sample-hz", type=float, default=10.0)
    parser.add_argument("--imu-bus", type=int, default=runtime_config.DEFAULT_IMU_BUS)
    parser.add_argument(
        "--imu-address", type=lambda v: int(v, 0),
        default=runtime_config.DEFAULT_IMU_ADDRESS,
    )
    parser.add_argument(
        "--imu-yaw-axis", choices=("gravity", "z"),
        default=runtime_config.DEFAULT_IMU_YAW_AXIS,
    )
    args = parser.parse_args(argv)

    print(
        "\n연속적인 소리(음악/물소리/선풍기)를 켜 두고, 소리와 몸은 그대로 둔 채\n"
        "키링만 좌우로 천천히 (+-60도 정도) 계속 돌리세요.\n",
        flush=True,
    )
    doa = DOAReader(poll_interval=0.05)
    if not doa.ok:
        print(f"DOA를 쓸 수 없습니다: {doa.describe()}", file=sys.stderr)
        return 1
    imu = IMUReader(
        bus_number=args.imu_bus,
        address=args.imu_address,
        yaw_axis=args.imu_yaw_axis,
    )
    if not imu.ok:
        print(f"IMU를 쓸 수 없습니다: {imu.describe()}", file=sys.stderr)
        doa.stop()
        return 1

    interval = 1.0 / max(1.0, args.sample_hz)
    yaws: list[float] = []
    angles: list[float] = []
    prev_yaw = prev_doa = None
    total_yaw = total_doa = 0.0
    skipped = 0

    print("측정 시작...", flush=True)
    deadline = time.monotonic() + args.seconds
    while time.monotonic() < deadline:
        time.sleep(interval)
        reading = doa.snapshot()
        if reading.raw_angle is None:
            skipped += 1
            continue
        swing = imu.swing_at(reading.timestamp, max_age=0.3)
        state = imu.snapshot()
        if swing is None:
            skipped += 1
            continue
        total_yaw = _unwrapped(prev_yaw, total_yaw, state.yaw)
        total_doa = _unwrapped(prev_doa, total_doa, float(reading.raw_angle))
        prev_yaw, prev_doa = state.yaw, float(reading.raw_angle)
        yaws.append(total_yaw)
        angles.append(total_doa)
        remaining = deadline - time.monotonic()
        print(
            f"\r  남은 시간 {remaining:4.1f}s | yaw={total_yaw:+7.1f} "
            f"doa={reading.raw_angle:3d} 표본={len(yaws):4d}",
            end="",
            flush=True,
        )

    print()
    doa.stop()
    imu.stop()

    if len(yaws) < 20:
        print(
            f"표본이 부족합니다 ({len(yaws)}개, 건너뜀 {skipped}개). "
            "소리가 계속 나고 있는지 확인하세요.",
            file=sys.stderr,
        )
        return 1

    span = max(yaws) - min(yaws)
    if span < MIN_YAW_SPAN_DEG:
        print(
            f"회전량이 너무 작습니다 (yaw 범위 {span:.1f}도 < "
            f"{MIN_YAW_SPAN_DEG:.0f}도). 더 크게 돌리고 다시 실행하세요.",
            file=sys.stderr,
        )
        return 1

    count = float(len(yaws))
    mean_yaw = sum(yaws) / count
    mean_doa = sum(angles) / count
    covariance = sum(
        (y - mean_yaw) * (a - mean_doa) for y, a in zip(yaws, angles)
    )
    variance_yaw = sum((y - mean_yaw) ** 2 for y in yaws)
    variance_doa = sum((a - mean_doa) ** 2 for a in angles)
    slope = covariance / variance_yaw
    r_squared = (
        0.0 if variance_doa <= 0 else covariance**2 / (variance_yaw * variance_doa)
    )

    print(f"\n표본 {len(yaws)}개 (건너뜀 {skipped}), yaw 범위 {span:.1f}도")
    print(f"기울기 d(DOA)/d(yaw) = {slope:+.3f},  R^2 = {r_squared:.3f}")

    if r_squared < MIN_R_SQUARED or not (SLOPE_MIN <= abs(slope) <= SLOPE_MAX):
        print(
            "\n판정 불가: DOA가 회전을 따라가지 않았습니다.\n"
            "소리가 끊기지 않는지, 소리와 몸이 고정돼 있는지 확인하고 다시 실행하세요.",
            file=sys.stderr,
        )
        return 1

    sign = -1.0 if slope > 0 else 1.0
    print(f"\n==> --imu-swing-sign {sign:+.0f}")
    if sign == runtime_config.DEFAULT_IMU_SWING_SIGN:
        print("    (runtime_config.DEFAULT_IMU_SWING_SIGN 기본값과 같습니다. 그대로 두세요.)")
    else:
        print(
            f"    기본값은 {runtime_config.DEFAULT_IMU_SWING_SIGN:+.0f}입니다. "
            f"runtime_config.DEFAULT_IMU_SWING_SIGN을 {sign:+.0f}로 바꾸거나\n"
            f"    main.py 실행 시 --imu-swing-sign {sign:+.0f}을 붙이세요."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
