"""IMU 샘플 구간을 움직임 상태로 요약한다.

키링 형태 기기라 사용자가 손으로 만지거나 걸을 때 옷/손 마찰음이 마이크에
그대로 들어온다. 오디오 윈도우와 같은 구간의 움직임을 함께 보내면 앱이 그런
구간의 감지를 걸러낼 수 있고, 충격/낙하처럼 소리만으로는 알 수 없는 상황도
같이 알릴 수 있다.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import atan2, degrees
from typing import Sequence

import numpy as np

from imu_module import ImuSample

STATE_UNKNOWN = "unknown"
STATE_STILL = "still"
STATE_MOTION = "motion"
STATE_SHOCK = "shock"
STATE_FREE_FALL = "free_fall"

# 정지 판정 상한. 정지 상태 노이즈는 보통 0.01g / 1dps 수준이라 여유를 뒀다.
STILL_ACCEL_RMS_G = 0.04
STILL_GYRO_MAX_DPS = 12.0
# 손으로 흔드는 정도는 여기까지, 그 이상은 충격으로 본다.
SHOCK_ACCEL_PEAK_G = 2.2
SHOCK_GYRO_PEAK_DPS = 400.0
# 자유낙하 중에는 합성 가속도가 0에 가까워진다.
FREE_FALL_ACCEL_G = 0.35
FREE_FALL_MIN_SECONDS = 0.08


@dataclass(frozen=True)
class MotionSnapshot:
    """오디오 윈도우 한 개에 대응하는 움직임 요약."""

    state: str
    sample_count: int
    accel_rms_g: float
    accel_peak_g: float
    gyro_max_dps: float
    pitch_deg: float
    roll_deg: float
    temperature_c: float

    @property
    def is_known(self) -> bool:
        return self.state != STATE_UNKNOWN

    @property
    def is_disturbed(self) -> bool:
        """정지가 아닌, 소리에 움직임이 섞였을 구간인지."""
        return self.state in (STATE_MOTION, STATE_SHOCK, STATE_FREE_FALL)

    @property
    def is_impact(self) -> bool:
        """소리와 별개로 앱에 알릴 만한 충격/낙하인지."""
        return self.state in (STATE_SHOCK, STATE_FREE_FALL)

    def as_dict(self) -> dict:
        return {
            "state": self.state,
            "samples": self.sample_count,
            "accel_rms_g": round(self.accel_rms_g, 4),
            "accel_peak_g": round(self.accel_peak_g, 4),
            "gyro_max_dps": round(self.gyro_max_dps, 2),
            "pitch_deg": round(self.pitch_deg, 1),
            "roll_deg": round(self.roll_deg, 1),
            "temperature_c": round(self.temperature_c, 1),
        }

    def summary(self) -> str:
        if not self.is_known:
            return "motion=unknown"
        return (
            f"motion={self.state} "
            f"a_rms={self.accel_rms_g:.3f}g "
            f"g_max={self.gyro_max_dps:.0f}dps"
        )


UNKNOWN_SNAPSHOT = MotionSnapshot(
    state=STATE_UNKNOWN,
    sample_count=0,
    accel_rms_g=0.0,
    accel_peak_g=0.0,
    gyro_max_dps=0.0,
    pitch_deg=0.0,
    roll_deg=0.0,
    temperature_c=0.0,
)


def summarize_motion(
    samples: Sequence[ImuSample],
    *,
    still_accel_rms_g: float = STILL_ACCEL_RMS_G,
    still_gyro_max_dps: float = STILL_GYRO_MAX_DPS,
) -> MotionSnapshot:
    """샘플 구간을 MotionSnapshot으로 축약한다. 샘플이 없으면 unknown."""
    if len(samples) < 2:
        return UNKNOWN_SNAPSHOT

    accel = np.asarray([sample.accel_g for sample in samples], dtype=np.float64)
    gyro = np.asarray([sample.gyro_dps for sample in samples], dtype=np.float64)

    magnitude = np.linalg.norm(accel, axis=1)
    # 중력 1g를 뺀 나머지가 실제 움직임 성분이다.
    linear = magnitude - 1.0
    accel_rms = float(np.sqrt(np.mean(np.square(linear))))
    accel_peak = float(np.max(magnitude))
    gyro_max = float(np.max(np.linalg.norm(gyro, axis=1)))

    # 자세는 구간 평균으로 낸다. 흔들리는 중이면 어차피 참고값이다.
    mean_accel = accel.mean(axis=0)
    ax, ay, az = mean_accel
    pitch = degrees(atan2(-ax, max(1e-9, (ay * ay + az * az) ** 0.5)))
    roll = degrees(atan2(ay, az))
    temperature = float(np.mean([sample.temperature_c for sample in samples]))

    state = _classify(
        samples=samples,
        magnitude=magnitude,
        accel_rms=accel_rms,
        accel_peak=accel_peak,
        gyro_max=gyro_max,
        still_accel_rms_g=still_accel_rms_g,
        still_gyro_max_dps=still_gyro_max_dps,
    )
    return MotionSnapshot(
        state=state,
        sample_count=len(samples),
        accel_rms_g=accel_rms,
        accel_peak_g=accel_peak,
        gyro_max_dps=gyro_max,
        pitch_deg=pitch,
        roll_deg=roll,
        temperature_c=temperature,
    )


def _classify(
    *,
    samples: Sequence[ImuSample],
    magnitude: np.ndarray,
    accel_rms: float,
    accel_peak: float,
    gyro_max: float,
    still_accel_rms_g: float,
    still_gyro_max_dps: float,
) -> str:
    if _free_fall_seconds(samples, magnitude) >= FREE_FALL_MIN_SECONDS:
        return STATE_FREE_FALL
    if accel_peak >= SHOCK_ACCEL_PEAK_G or gyro_max >= SHOCK_GYRO_PEAK_DPS:
        return STATE_SHOCK
    if accel_rms <= still_accel_rms_g and gyro_max <= still_gyro_max_dps:
        return STATE_STILL
    return STATE_MOTION


class MotionMonitor:
    """오디오 윈도우 한 개에 대응하는 움직임 요약을 제공한다.

    컨트롤러가 ImuModule을 직접 알지 않도록 감싸는 얇은 층이다. IMU가 없거나
    통신이 끊기면 unknown을 돌려주므로 호출부는 분기 없이 그대로 쓴다.
    """

    def __init__(
        self,
        imu: object | None,
        *,
        window_seconds: float,
        suppress_on_motion: bool = False,
    ) -> None:
        self.imu = imu
        self.window_seconds = max(0.1, float(window_seconds))
        self.suppress_on_motion = bool(suppress_on_motion)

    @property
    def enabled(self) -> bool:
        return self.imu is not None

    def snapshot(self) -> MotionSnapshot:
        """직전 window_seconds 구간의 움직임 요약.

        오디오 윈도우의 캡처 시각이 아니라 '지금'을 기준으로 뒤로 자른다.
        윈도우가 채워지자마자 호출되므로 평시에는 추론 지연(수백 ms)만큼만
        어긋나지만, 큐가 밀려 윈도우가 몰아서 처리되면 최대 hop 몇 개만큼
        틀어질 수 있다. 정확히 맞추려면 윈도우에 캡처 시각을 실어야 한다.
        """
        if self.imu is None or not getattr(self.imu, "healthy", True):
            return UNKNOWN_SNAPSHOT
        return summarize_motion(self.imu.recent_samples(self.window_seconds))

    def should_suppress(self, snapshot: MotionSnapshot) -> bool:
        """기기를 만지는 중이라 마찰음이 섞였을 구간인지.

        shock과 free_fall은 일부러 제외한다. 유리가 깨지거나 물건이 떨어지는
        순간이 바로 그 상태이고, 그때가 glass_shatter를 잡아야 할 때다.
        오탐 몇 개보다 미탐 하나가 훨씬 나쁘므로 손으로 다루는 동안의
        마찰음(motion)만 막는다.
        """
        return self.suppress_on_motion and snapshot.state == STATE_MOTION


def _free_fall_seconds(samples: Sequence[ImuSample], magnitude: np.ndarray) -> float:
    """합성 가속도가 임계 아래로 연속해서 머문 최장 시간."""
    below = magnitude < FREE_FALL_ACCEL_G
    if not below.any():
        return 0.0

    longest = 0.0
    run_start: float | None = None
    for index, is_below in enumerate(below):
        if is_below:
            if run_start is None:
                run_start = samples[index].monotonic
            longest = max(longest, samples[index].monotonic - run_start)
        else:
            run_start = None
    return longest
