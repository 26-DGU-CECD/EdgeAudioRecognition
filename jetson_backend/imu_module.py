"""MPU9250/MPU6500 IMU 드라이버와 백그라운드 샘플러.

I2C 버스에서 가속도/자이로를 주기적으로 읽어 최근 구간을 링 버퍼에 유지한다.
오디오 윈도우 한 개를 처리할 때마다 같은 구간의 움직임 통계를 뽑아 쓰는 것이
목적이므로, 샘플러는 오디오 스레드와 독립적으로 동작한다.
"""
from __future__ import annotations

import json
import struct
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Deque, List

import numpy as np

# --- MPU9250 레지스터 ---
REG_SMPLRT_DIV = 0x19
REG_CONFIG = 0x1A
REG_GYRO_CONFIG = 0x1B
REG_ACCEL_CONFIG = 0x1C
REG_ACCEL_CONFIG2 = 0x1D
REG_INT_PIN_CFG = 0x37
REG_ACCEL_XOUT_H = 0x3B
REG_USER_CTRL = 0x6A
REG_PWR_MGMT_1 = 0x6B
REG_PWR_MGMT_2 = 0x6C
REG_WHO_AM_I = 0x75

# WHO_AM_I 값 -> 칩 이름. 모두 동일한 레지스터 맵을 쓴다.
KNOWN_CHIPS = {
    0x68: "MPU6050",
    0x70: "MPU6500",
    0x71: "MPU9250",
    0x73: "MPU9255",
    0x75: "MPU9250(rev)",
}

ACCEL_FS_G = 2.0
GYRO_FS_DPS = 250.0
ACCEL_SCALE = ACCEL_FS_G / 32768.0
GYRO_SCALE = GYRO_FS_DPS / 32768.0
TEMP_SENSITIVITY = 333.87
TEMP_OFFSET_C = 21.0

DEFAULT_BUS = 1
DEFAULT_ADDRESS = 0x68
DEFAULT_SAMPLE_HZ = 50.0
DEFAULT_HISTORY_SECONDS = 4.0
CALIBRATION_FILE = Path(__file__).resolve().parent / "imu_calibration.json"


@dataclass(frozen=True)
class ImuSample:
    monotonic: float
    accel_g: tuple[float, float, float]
    gyro_dps: tuple[float, float, float]
    temperature_c: float


class ImuError(RuntimeError):
    """IMU 초기화/통신 실패."""


@dataclass
class ImuCalibration:
    """가속도/자이로 보정값. 가속도는 g, 자이로는 dps 단위.

    적용 순서는 ``(측정값 - bias) * accel_scale``이다. accel_bias는 6면
    캘리브레이션으로만 제대로 구할 수 있어 기본값은 0이고, 자동 측정은
    accel_scale과 gyro_bias만 채운다(:meth:`ImuModule.calibrate` 참고).
    """

    accel_bias: tuple[float, float, float] = (0.0, 0.0, 0.0)
    gyro_bias: tuple[float, float, float] = (0.0, 0.0, 0.0)
    accel_scale: float = 1.0

    @classmethod
    def load(cls, path: Path | str = CALIBRATION_FILE) -> "ImuCalibration":
        calibration_path = Path(path)
        if not calibration_path.is_file():
            return cls()
        try:
            with calibration_path.open(encoding="utf-8") as handle:
                document = json.load(handle)
            return cls(
                accel_bias=tuple(float(v) for v in document["accel_bias"]),
                gyro_bias=tuple(float(v) for v in document["gyro_bias"]),
                accel_scale=float(document.get("accel_scale", 1.0)),
            )
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            return cls()

    def save(self, path: Path | str = CALIBRATION_FILE) -> Path:
        calibration_path = Path(path)
        document = {
            "accel_bias": list(self.accel_bias),
            "gyro_bias": list(self.gyro_bias),
            "accel_scale": self.accel_scale,
        }
        with calibration_path.open("w", encoding="utf-8") as handle:
            json.dump(document, handle, indent=2)
        return calibration_path


class Mpu9250:
    """smbus2 기반 저수준 드라이버."""

    def __init__(
        self,
        *,
        bus: int = DEFAULT_BUS,
        address: int = DEFAULT_ADDRESS,
        calibration: ImuCalibration | None = None,
    ) -> None:
        try:
            from smbus2 import SMBus
        except ImportError as exc:  # pragma: no cover - 하드웨어 없는 환경
            raise ImuError(
                f"smbus2를 불러올 수 없습니다: {exc}. "
                "pip install smbus2 후 다시 시도하세요."
            ) from exc

        self.bus_number = int(bus)
        self.address = int(address)
        self.calibration = calibration or ImuCalibration()
        # 샘플러 스레드와 calibrate()가 같은 버스를 쓰므로 직렬화한다.
        self._io_lock = threading.Lock()
        try:
            self._bus = SMBus(self.bus_number)
        except OSError as exc:
            raise ImuError(
                f"I2C 버스 {self.bus_number}를 열 수 없습니다: {exc}. "
                "raspi-config에서 I2C를 활성화했는지 확인하세요."
            ) from exc

        self.chip_name = self._identify()
        self._configure()

    def _identify(self) -> str:
        try:
            who_am_i = self._bus.read_byte_data(self.address, REG_WHO_AM_I)
        except OSError as exc:
            self.close()
            raise ImuError(
                f"주소 0x{self.address:02X}에서 응답이 없습니다: {exc}. "
                f"`i2cdetect -y {self.bus_number}`로 연결을 확인하세요."
            ) from exc
        if who_am_i not in KNOWN_CHIPS:
            self.close()
            raise ImuError(
                f"알 수 없는 WHO_AM_I=0x{who_am_i:02X}. "
                "MPU6050/6500/9250 계열이 아닙니다."
            )
        return KNOWN_CHIPS[who_am_i]

    def _configure(self) -> None:
        try:
            self._bus.write_byte_data(self.address, REG_PWR_MGMT_1, 0x80)  # reset
            time.sleep(0.1)
            self._bus.write_byte_data(self.address, REG_PWR_MGMT_1, 0x01)  # PLL
            time.sleep(0.05)
            self._bus.write_byte_data(self.address, REG_PWR_MGMT_2, 0x00)  # 전 축 on
            self._bus.write_byte_data(self.address, REG_CONFIG, 0x03)  # DLPF 41Hz
            self._bus.write_byte_data(self.address, REG_SMPLRT_DIV, 0x04)  # 200Hz
            self._bus.write_byte_data(self.address, REG_GYRO_CONFIG, 0x00)  # ±250dps
            self._bus.write_byte_data(self.address, REG_ACCEL_CONFIG, 0x00)  # ±2g
            self._bus.write_byte_data(self.address, REG_ACCEL_CONFIG2, 0x03)
            self._bus.write_byte_data(self.address, REG_USER_CTRL, 0x00)
            self._bus.write_byte_data(self.address, REG_INT_PIN_CFG, 0x02)  # bypass
            time.sleep(0.05)
        except OSError as exc:
            self.close()
            raise ImuError(f"IMU 설정 쓰기 실패: {exc}") from exc

    def read_sample(self) -> ImuSample:
        with self._io_lock:
            raw = self._bus.read_i2c_block_data(self.address, REG_ACCEL_XOUT_H, 14)
        ax, ay, az, temp, gx, gy, gz = struct.unpack(">hhhhhhh", bytes(raw))
        bias_a = self.calibration.accel_bias
        bias_g = self.calibration.gyro_bias
        scale_a = self.calibration.accel_scale
        return ImuSample(
            monotonic=time.monotonic(),
            accel_g=(
                (ax * ACCEL_SCALE - bias_a[0]) * scale_a,
                (ay * ACCEL_SCALE - bias_a[1]) * scale_a,
                (az * ACCEL_SCALE - bias_a[2]) * scale_a,
            ),
            gyro_dps=(
                gx * GYRO_SCALE - bias_g[0],
                gy * GYRO_SCALE - bias_g[1],
                gz * GYRO_SCALE - bias_g[2],
            ),
            temperature_c=temp / TEMP_SENSITIVITY + TEMP_OFFSET_C,
        )

    def read_raw_sample(self) -> ImuSample:
        """캘리브레이션을 적용하지 않은 값. 바이어스 측정에 쓴다."""
        saved = self.calibration
        self.calibration = ImuCalibration()
        try:
            return self.read_sample()
        finally:
            self.calibration = saved

    def close(self) -> None:
        bus = getattr(self, "_bus", None)
        if bus is not None:
            try:
                bus.close()
            except OSError:
                pass
            self._bus = None


class ImuModule:
    """IMU를 백그라운드에서 폴링하고 최근 구간 샘플을 제공한다."""

    def __init__(
        self,
        *,
        bus: int = DEFAULT_BUS,
        address: int = DEFAULT_ADDRESS,
        sample_hz: float = DEFAULT_SAMPLE_HZ,
        history_seconds: float = DEFAULT_HISTORY_SECONDS,
        calibration: ImuCalibration | None = None,
    ) -> None:
        self.sample_hz = max(1.0, float(sample_hz))
        self.history_seconds = max(0.5, float(history_seconds))
        self._period = 1.0 / self.sample_hz
        self._sensor = Mpu9250(bus=bus, address=address, calibration=calibration)
        self._samples: Deque[ImuSample] = deque(
            maxlen=int(self.sample_hz * self.history_seconds) + 8
        )
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self.read_errors = 0
        self._consecutive_errors = 0
        self.healthy = True

    @property
    def chip_name(self) -> str:
        return self._sensor.chip_name

    @property
    def description(self) -> str:
        return (
            f"{self.chip_name} @ i2c-{self._sensor.bus_number}"
            f":0x{self._sensor.address:02X} {self.sample_hz:.0f}Hz"
        )

    def start(self) -> None:
        if self._thread is not None:
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run,
            name="imu-sampler",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        self._sensor.close()

    def __enter__(self) -> "ImuModule":
        self.start()
        return self

    def __exit__(self, *_exc_info: object) -> None:
        self.stop()

    def _run(self) -> None:
        next_at = time.monotonic()
        while not self._stop_event.is_set():
            try:
                sample = self._sensor.read_sample()
            except OSError:
                self.read_errors += 1
                self._consecutive_errors += 1
                # 버스 글리치 몇 번은 넘기고, 계속 실패하면 비정상으로 표시한다.
                if self._consecutive_errors >= 10:
                    self.healthy = False
            else:
                self._consecutive_errors = 0
                self.healthy = True
                with self._lock:
                    self._samples.append(sample)

            next_at += self._period
            delay = next_at - time.monotonic()
            if delay < 0.0:
                next_at = time.monotonic()
            else:
                self._stop_event.wait(delay)

    def recent_samples(self, seconds: float) -> List[ImuSample]:
        cutoff = time.monotonic() - max(0.0, float(seconds))
        with self._lock:
            return [sample for sample in self._samples if sample.monotonic >= cutoff]

    def latest_sample(self) -> ImuSample | None:
        with self._lock:
            return self._samples[-1] if self._samples else None

    def calibrate(self, seconds: float = 2.0) -> ImuCalibration:
        """정지 상태에서 호출한다. 자세는 어떻든 상관없다.

        한 자세만 재서는 축별 가속도 바이어스와 실제 기울기를 분리할 수 없으므로
        축별 바이어스는 건드리지 않는다. 대신 정지 시 합성 가속도 크기가 1g가
        되도록 스케일만 맞추고, 정지 중에는 0이어야 하는 자이로 바이어스를 뺀다.
        움직임 판정은 ``|합성가속도 - 1g|``와 자이로만 보므로 이 둘이면 충분하다.
        """
        deadline = time.monotonic() + max(0.5, float(seconds))
        accel_rows: list[tuple[float, float, float]] = []
        gyro_rows: list[tuple[float, float, float]] = []
        while time.monotonic() < deadline:
            try:
                sample = self._sensor.read_raw_sample()
            except OSError:
                continue
            accel_rows.append(sample.accel_g)
            gyro_rows.append(sample.gyro_dps)
            time.sleep(self._period)

        if len(accel_rows) < 10:
            raise ImuError("캘리브레이션 표본이 부족합니다. 연결을 확인하세요.")

        accel_mean = np.asarray(accel_rows, dtype=np.float64).mean(axis=0)
        gyro_mean = np.asarray(gyro_rows, dtype=np.float64).mean(axis=0)

        # 정지 중이면 합성 가속도는 자세와 무관하게 1g여야 한다.
        accel_bias = np.asarray(self._sensor.calibration.accel_bias, dtype=np.float64)
        magnitude = float(np.linalg.norm(accel_mean - accel_bias))
        if not 0.5 <= magnitude <= 1.6:
            raise ImuError(
                f"정지 상태 가속도 크기가 {magnitude:.3f}g로 비정상입니다. "
                "센서를 움직이지 말고 다시 시도하세요."
            )

        calibration = ImuCalibration(
            accel_bias=self._sensor.calibration.accel_bias,
            gyro_bias=tuple(float(v) for v in gyro_mean),
            accel_scale=float(1.0 / magnitude),
        )
        self._sensor.calibration = calibration
        return calibration


def open_imu(
    *,
    bus: int = DEFAULT_BUS,
    address: int = DEFAULT_ADDRESS,
    sample_hz: float = DEFAULT_SAMPLE_HZ,
    history_seconds: float = DEFAULT_HISTORY_SECONDS,
    calibration_path: Path | str = CALIBRATION_FILE,
) -> ImuModule:
    """저장된 캘리브레이션을 적용해 IMU를 열고 샘플링을 시작한다."""
    module = ImuModule(
        bus=bus,
        address=address,
        sample_hz=sample_hz,
        history_seconds=history_seconds,
        calibration=ImuCalibration.load(calibration_path),
    )
    module.start()
    return module
