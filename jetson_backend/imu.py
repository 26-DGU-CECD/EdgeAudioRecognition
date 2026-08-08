"""MPU-9250 yaw de-swing for the ReSpeaker DSP direction-of-arrival angle.

The keyring swings while it hangs, so the DSP angle — measured in the device
frame — swings with it even when the sound source is fixed. This module tracks
how far the device has rotated away from its own slowly-tracked reference
heading (`swing`) so the publish path can add it back out.

Deliberate non-goals, all of them load-bearing:

- **No absolute heading.** Gyro integration drifts without bound, but `swing`
  is a difference of two quantities that drift together (`yaw` and `ref`), so
  the drift cancels. Nothing here means "north".
- **No magnetometer (AK8963).** It is never initialized or read. Indoors it is
  dominated by the speaker magnets and the Pi itself.
- **No tilt compensation of the DOA angle.** The DSP's DOA is a 2D algorithm
  that discards elevation, so tilt-correcting it would be an approximation on
  top of an approximation. Accelerometer data is used for exactly one thing:
  deciding *which* rotation axis is yaw (see `yaw_axis` below).
- **No pip IMU library.** Registers are read directly over `smbus2`, which is
  already a dependency (`battery_monitor.py`).

Shake versus turn
-----------------
A hanging keyring oscillates at roughly 0.5-2 Hz; a wearer turning takes 1-2
seconds. Those bands overlap, so a lowpass on the angle cannot separate them.
What *does* separate them is sign structure: an oscillation keeps reversing,
a turn does not.

The obvious form of that test — "is the mean yaw rate over the last half second
large?" — does not work, and it fails in the worst possible direction. Averaging
a sinusoid over a fraction of its own period does not cancel it, it rectifies
it. Over a window T the worst-case mean of an amplitude-A oscillation at
frequency f is `2*A*|sin(pi*f*T)| / T`, so a +-25 deg shake measured over 0.5 s
peaks at:

    0.5 Hz -> 71 deg/s   0.75 Hz -> 92 deg/s   1.0 Hz -> 100 deg/s

against only 60 deg/s for a real 90-degree turn taking 1.5 s. The shake scores
*higher* than the turn, so a plain mean gate marks a swinging keyring as
"turning", lets the reference heading chase the swing, and cancels the very
correction this module exists to produce.

Lengthening the window alone does not rescue it either: at 4 s the shake nulls
out but the same turn falls to 22.5 deg/s, back under the threshold. The window
would have to be an exact multiple of a shake period that is not known in
advance.

So the magnitude test is kept but paired with a shape test — and the shape test
counts *reversals*, not sub-window averages:

    |mean(yaw_rate, W)| > turn_threshold
      AND the yaw rate never reversed within W   ->  turning
    otherwise                                    ->  shaking

Averages were tried first and alias. Near an oscillation's peak the rate is
small, so a slice straddling the peak still averages to the pre-peak sign, and
at high frequency whole slices alias into agreement; measured leaks appeared at
0.4-0.5 Hz and 2-3 Hz. A reversal count cannot alias: an oscillation at f Hz
produces 2f reversals per second whatever its amplitude or phase, so every
swing whose half period fits inside W is caught exactly, while a turn produces
none. The default W of 1.2 s therefore rejects everything above 1/(2W) = 0.42
Hz, covering the 0.5-2 Hz swing band with margin, and costs at most 1.2 s of
latency in noticing a real turn.

`swing = wrap(yaw - ref)` is therefore ~the shake during a shake, and decays to
~0 shortly after a real turn, which is what preserves genuine direction changes.
"""
from __future__ import annotations

import math
import struct
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass

MPU9250_ADDRESS = 0x68

# Register map (MPU-9250 datasheet rev 1.6). Only the accel/gyro half is used;
# the AK8963 magnetometer behind the aux I2C bus is never touched.
REG_SMPLRT_DIV = 0x19
REG_CONFIG = 0x1A
REG_GYRO_CONFIG = 0x1B
REG_ACCEL_CONFIG = 0x1C
REG_ACCEL_CONFIG2 = 0x1D
REG_ACCEL_XOUT_H = 0x3B
REG_PWR_MGMT_1 = 0x6B
REG_PWR_MGMT_2 = 0x6C
REG_WHO_AM_I = 0x75

# 0x71 is the genuine MPU-9250; 0x73 is the MPU-9255 sibling; 0x68 shows up on
# some clones that are otherwise register-compatible.
WHO_AM_I_EXPECTED = (0x71, 0x73, 0x68)

# GYRO_CONFIG FS_SEL -> (register bits, LSB per deg/s). +-500 dps is the default
# because a brisk keyring shake clips +-250 dps, and a clipped sample corrupts
# the yaw integral for good.
GYRO_RANGES = {
    250: (0x00, 131.0),
    500: (0x08, 65.5),
    1000: (0x10, 32.8),
    2000: (0x18, 16.4),
}
ACCEL_LSB_PER_G = 16384.0  # ACCEL_CONFIG = 0x00 -> +-2 g

# Bias calibration is meant to run while the unit is still. If it is not, the
# bias soaks up real motion and every later reading is skewed, so warn loudly.
CALIBRATION_STD_WARN_DPS = 2.0

# Hysteresis band for counting yaw-rate reversals. The rate has to swing past
# +-this to count as having changed direction, so sensor noise around zero while
# the unit sits still cannot manufacture reversals. A real turn runs far outside
# the band, so the band never suppresses a genuine one.
TURN_REVERSAL_EPSILON_DPS = 2.0
# Keep chasing at the fast time constant briefly after the gate drops. Without
# it the last few degrees of a turn are left to decay at the slow still-state
# constant, parking a several-degree bias in `swing` after every turn.
TURN_HOLD_SECONDS = 0.5

# Below this the "still" branch also nudges the gravity axis, so a keyring that
# settles at a different hang angle than it was calibrated at re-learns its yaw
# axis instead of silently measuring the wrong rotation.
UP_UPDATE_GYRO_DPS = 5.0
UP_UPDATE_TAU_SECONDS = 10.0
UP_UPDATE_GRAVITY_TOLERANCE_G = 0.15


def wrap180(degrees: float) -> float:
    """Shortest signed angle in [-180, 180). 359 and 1 are 2 degrees apart."""
    return (float(degrees) + 180.0) % 360.0 - 180.0


@dataclass(frozen=True)
class IMUReading:
    """One integrated sample. Angles are degrees, rates degrees/second."""

    timestamp: float
    yaw: float
    ref: float
    swing: float
    yaw_rate: float
    gyro_mean: float
    turning: bool


DISABLED_IMU_READING = IMUReading(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, False)


class IMUReader:
    """Background poller that maintains `swing` and answers `swing_at(t)`.

    Same shape as `DOAReader`: constructor probes the hardware and never raises,
    `describe()` reports why it is off, a daemon thread does the polling, and
    the publish path only ever reads snapshots.
    """

    def __init__(
        self,
        *,
        enabled: bool = True,
        bus_number: int = 1,
        address: int = MPU9250_ADDRESS,
        poll_hz: float = 100.0,
        turn_threshold: float = 10.0,
        turn_window_seconds: float = 1.2,
        ref_tau_still: float = 4.0,
        ref_tau_turn: float = 0.25,
        calibration_seconds: float = 1.5,
        gyro_range_dps: int = 500,
        yaw_axis: str = "gravity",
        history_seconds: float = 5.0,
    ) -> None:
        self.ok = False
        self.status = "disabled"
        self.address = int(address)
        self.bus_number = int(bus_number)
        self.poll_hz = min(400.0, max(10.0, float(poll_hz)))
        self.poll_interval = 1.0 / self.poll_hz
        self.turn_threshold = abs(float(turn_threshold))
        self.turn_window_seconds = max(0.05, float(turn_window_seconds))
        self.ref_tau_still = max(0.05, float(ref_tau_still))
        self.ref_tau_turn = max(0.01, float(ref_tau_turn))
        self.calibration_seconds = max(0.2, float(calibration_seconds))
        self.yaw_axis = yaw_axis if yaw_axis in ("gravity", "z") else "gravity"

        self._bus = None
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._last_error: str | None = None
        self._error_streak = 0

        self._gyro_bias = (0.0, 0.0, 0.0)
        self._gyro_bias_std = 0.0
        # Accelerometer at rest measures specific force, i.e. it points UP. With
        # the board flat and its Z axis up this is (0, 0, 1) and the projection
        # below degenerates to plain gyro Z, exactly as the flat-mount case
        # expects; it just keeps working when the board is not flat.
        self._up = (0.0, 0.0, 1.0)
        self._accel = (0.0, 0.0, 0.0)

        self._yaw = 0.0
        self._ref = 0.0
        self._swing = 0.0
        self._yaw_rate = 0.0
        self._gyro_mean = 0.0
        self._turning = False
        self._turn_until = 0.0
        self._last_sample_at: float | None = None

        self._rate_window: deque[tuple[float, float]] = deque()
        self._reversals: deque[float] = deque()
        self._rate_sign = 0
        history = max(1.0, float(history_seconds))
        self._history: deque[IMUReading] = deque(
            maxlen=int(history * self.poll_hz) + 8
        )

        if not enabled:
            print("[IMU] --disable-imu: 흔들림 보정을 끕니다.", file=sys.stderr, flush=True)
            return

        gyro_bits, self._gyro_lsb = GYRO_RANGES.get(
            int(gyro_range_dps), GYRO_RANGES[500]
        )
        try:
            self._open(gyro_bits)
        except Exception as exc:
            self.status = f"disabled:{type(exc).__name__}"
            self._last_error = str(exc)
            self._close_bus()
            hint = ""
            if isinstance(exc, OSError):
                hint = (
                    f"\n[IMU] i2cdetect -y {self.bus_number} 로 "
                    f"0x{self.address:02x} 가 보이는지 확인하세요."
                )
            print(f"[IMU] 비활성화: {exc!r}{hint}", file=sys.stderr, flush=True)
            return

        self.ok = True
        self.status = "enabled"
        self._thread = threading.Thread(
            target=self._poll_loop, name="imu-poll", daemon=True
        )
        self._thread.start()

    # ------------------------------------------------------------------ setup

    def _open(self, gyro_bits: int) -> None:
        import smbus2

        self._bus = smbus2.SMBus(self.bus_number)
        who = self._bus.read_byte_data(self.address, REG_WHO_AM_I)
        if who not in WHO_AM_I_EXPECTED:
            raise RuntimeError(
                f"WHO_AM_I=0x{who:02x} (기대값 "
                f"{', '.join(f'0x{v:02x}' for v in WHO_AM_I_EXPECTED)})"
            )

        self._bus.write_byte_data(self.address, REG_PWR_MGMT_1, 0x80)  # reset
        time.sleep(0.12)
        self._bus.write_byte_data(self.address, REG_PWR_MGMT_1, 0x01)  # gyro PLL
        time.sleep(0.02)
        self._bus.write_byte_data(self.address, REG_PWR_MGMT_2, 0x00)
        # DLPF 41 Hz on both sensors: above the 0.5-2 Hz swing band we care
        # about, below the sample rate, and it kills the structural buzz that
        # would otherwise integrate into yaw.
        self._bus.write_byte_data(self.address, REG_CONFIG, 0x03)
        self._bus.write_byte_data(self.address, REG_SMPLRT_DIV, 0x04)  # 200 Hz
        self._bus.write_byte_data(self.address, REG_GYRO_CONFIG, gyro_bits)
        self._bus.write_byte_data(self.address, REG_ACCEL_CONFIG, 0x00)
        self._bus.write_byte_data(self.address, REG_ACCEL_CONFIG2, 0x03)
        time.sleep(0.1)

        self._calibrate()

    def _calibrate(self) -> None:
        """Average the gyro at rest to get its bias, and find which way is up.

        The full 3-axis bias is stored, not just Z, because the yaw axis is a
        projection of the whole gyro vector.
        """
        samples: list[tuple[tuple[float, float, float], tuple[float, float, float]]] = []
        deadline = time.monotonic() + self.calibration_seconds
        while time.monotonic() < deadline:
            samples.append(self._read_raw())
            time.sleep(self.poll_interval)
        if len(samples) < 8:
            raise RuntimeError("바이어스 보정 표본이 너무 적습니다.")

        count = float(len(samples))
        self._gyro_bias = tuple(  # type: ignore[assignment]
            sum(sample[1][axis] for sample in samples) / count for axis in range(3)
        )
        accel_mean = tuple(
            sum(sample[0][axis] for sample in samples) / count for axis in range(3)
        )
        self._accel = accel_mean  # type: ignore[assignment]
        if self.yaw_axis == "gravity":
            self._up = _normalize(accel_mean) or (0.0, 0.0, 1.0)

        rates = [
            self._project(_subtract(sample[1], self._gyro_bias)) for sample in samples
        ]
        mean_rate = sum(rates) / count
        self._gyro_bias_std = math.sqrt(
            sum((rate - mean_rate) ** 2 for rate in rates) / count
        )

        axis_note = (
            f"up=({self._up[0]:+.2f},{self._up[1]:+.2f},{self._up[2]:+.2f})"
            if self.yaw_axis == "gravity"
            else "axis=gyro_z"
        )
        print(
            f"[IMU] 자이로 바이어스 "
            f"({self._gyro_bias[0]:+.3f},{self._gyro_bias[1]:+.3f},"
            f"{self._gyro_bias[2]:+.3f}) dps, "
            f"보정중 표준편차 {self._gyro_bias_std:.3f} dps, {axis_note}, "
            f"{len(samples)}표본/{self.calibration_seconds:.1f}s",
            file=sys.stderr,
            flush=True,
        )
        if self._gyro_bias_std > CALIBRATION_STD_WARN_DPS:
            print(
                f"[IMU] 경고: 보정 중 움직임이 감지되었습니다 "
                f"(표준편차 {self._gyro_bias_std:.2f} > "
                f"{CALIBRATION_STD_WARN_DPS:.1f} dps). "
                "바이어스가 부정확할 수 있습니다. 정지 상태에서 재시작하세요.",
                file=sys.stderr,
                flush=True,
            )
        if self.yaw_axis == "gravity" and abs(self._up[2]) < 0.5:
            print(
                f"[IMU] 참고: 칩 Z축이 수직에서 크게 벗어나 있습니다 "
                f"(up_z={self._up[2]:+.2f}). 중력축 투영으로 yaw를 계산합니다.",
                file=sys.stderr,
                flush=True,
            )

    # ----------------------------------------------------------------- device

    def _read_raw(
        self,
    ) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
        """One burst read -> (accel in g, gyro in deg/s), both unbiased-raw.

        Accel, temperature and gyro are contiguous, so a single 14-byte burst
        guarantees all six axes come from the same sample instant.
        """
        block = self._bus.read_i2c_block_data(self.address, REG_ACCEL_XOUT_H, 14)
        ax, ay, az, _temp, gx, gy, gz = struct.unpack(">7h", bytes(block))
        accel = (
            ax / ACCEL_LSB_PER_G,
            ay / ACCEL_LSB_PER_G,
            az / ACCEL_LSB_PER_G,
        )
        gyro = (
            gx / self._gyro_lsb,
            gy / self._gyro_lsb,
            gz / self._gyro_lsb,
        )
        return accel, gyro

    def _project(self, gyro: tuple[float, float, float]) -> float:
        """Rotation rate about the vertical, in degrees/second.

        With `yaw_axis="z"` this is literally gyro Z. With `"gravity"` it is the
        gyro vector projected onto the measured up direction, which is the same
        number when the board is flat and the correct one when it is not.
        """
        if self.yaw_axis == "z":
            return gyro[2]
        up = self._up
        return gyro[0] * up[0] + gyro[1] * up[1] + gyro[2] * up[2]

    # ------------------------------------------------------------------- loop

    def _poll_loop(self) -> None:
        next_at = time.monotonic()
        while not self._stop_event.is_set():
            next_at += self.poll_interval
            delay = next_at - time.monotonic()
            if delay > 0:
                if self._stop_event.wait(delay):
                    return
            else:
                # Fell behind (scheduler hiccup); resync instead of spinning.
                next_at = time.monotonic()
            self._poll_once()

    def _poll_once(self) -> None:
        try:
            accel, gyro_raw = self._read_raw()
        except OSError as exc:
            self._error_streak += 1
            self._last_error = type(exc).__name__
            if self._error_streak == 1:
                print(
                    f"[IMU] I2C 읽기 실패: {exc} | 보정 없이 계속합니다.",
                    file=sys.stderr,
                    flush=True,
                )
            if self._error_streak >= 5:
                self.status = f"i2c_error:{self._last_error}"
            return

        if self._error_streak:
            self._error_streak = 0
            self.status = "enabled"
        self._integrate(accel, gyro_raw, time.monotonic())

    def _integrate(
        self,
        accel: tuple[float, float, float],
        gyro_raw: tuple[float, float, float],
        now: float,
    ) -> IMUReading:
        """Advance yaw/ref/swing by one sample. Pure given (accel, gyro, now).

        Split out from the I2C read so the shake-versus-turn behaviour can be
        driven with synthetic motion without any hardware attached.
        """
        dt = 0.0 if self._last_sample_at is None else now - self._last_sample_at
        self._last_sample_at = now
        # A stalled thread must not dump a huge integration step into yaw.
        if dt <= 0.0 or dt > 0.5:
            dt = 0.0

        gyro = _subtract(gyro_raw, self._gyro_bias)
        yaw_rate = self._project(gyro)

        self._rate_window.append((now, yaw_rate))
        cutoff = now - self.turn_window_seconds
        while self._rate_window and self._rate_window[0][0] < cutoff:
            self._rate_window.popleft()
        gyro_mean = sum(rate for _t, rate in self._rate_window) / len(self._rate_window)

        self._note_reversal(yaw_rate, now)
        while self._reversals and self._reversals[0] < cutoff:
            self._reversals.popleft()
        # A window that is not yet full has not had the chance to show a
        # reversal, so it cannot prove a turn either.
        window_full = bool(self._rate_window) and self._rate_window[0][0] <= cutoff + (
            2.0 * self.poll_interval
        )
        gated = (
            abs(gyro_mean) > self.turn_threshold
            and not self._reversals
            and window_full
        )
        if gated:
            self._turn_until = now + TURN_HOLD_SECONDS
        turning = gated or now < self._turn_until

        yaw = (self._yaw + yaw_rate * dt) % 360.0
        tau = self.ref_tau_turn if turning else self.ref_tau_still
        alpha = 1.0 - math.exp(-dt / tau) if dt > 0.0 else 0.0
        ref = (self._ref + alpha * wrap180(yaw - self._ref)) % 360.0
        swing = wrap180(yaw - ref)

        if (
            self.yaw_axis == "gravity"
            and dt > 0.0
            and not turning
            and abs(gyro[0]) < UP_UPDATE_GYRO_DPS
            and abs(gyro[1]) < UP_UPDATE_GYRO_DPS
            and abs(gyro[2]) < UP_UPDATE_GYRO_DPS
            and abs(_norm(accel) - 1.0) < UP_UPDATE_GRAVITY_TOLERANCE_G
        ):
            beta = 1.0 - math.exp(-dt / UP_UPDATE_TAU_SECONDS)
            blended = _normalize(
                (
                    self._up[0] + beta * (accel[0] - self._up[0]),
                    self._up[1] + beta * (accel[1] - self._up[1]),
                    self._up[2] + beta * (accel[2] - self._up[2]),
                )
            )
            if blended is not None:
                self._up = blended

        reading = IMUReading(now, yaw, ref, swing, yaw_rate, gyro_mean, turning)
        with self._lock:
            self._accel = accel
            self._yaw = yaw
            self._ref = ref
            self._swing = swing
            self._yaw_rate = yaw_rate
            self._gyro_mean = gyro_mean
            self._turning = turning
            self._history.append(reading)
        return reading

    def _note_reversal(self, yaw_rate: float, now: float) -> None:
        """Record direction changes of the yaw rate, with hysteresis.

        This is the half of the discriminator the magnitude test cannot do.
        Counting reversals rather than comparing sub-window averages matters:
        averages alias. Near an oscillation's peak the rate is small, so a slice
        straddling the peak can still average to the pre-peak sign and let a
        pure shake pass as a turn, and at high frequency whole slices alias into
        agreement. A reversal count cannot alias — an oscillation at f Hz
        produces 2f of them per second no matter its amplitude or phase — so any
        swing whose half period fits inside the window is caught exactly.
        """
        if yaw_rate > TURN_REVERSAL_EPSILON_DPS:
            sign = 1
        elif yaw_rate < -TURN_REVERSAL_EPSILON_DPS:
            sign = -1
        else:
            return  # inside the band: no directional information, no update
        if self._rate_sign and sign != self._rate_sign:
            self._reversals.append(now)
        self._rate_sign = sign

    # ------------------------------------------------------------------ query

    def describe(self) -> str:
        return self.status

    def snapshot(self) -> IMUReading:
        """Most recent integrated sample, or a zeroed reading when disabled."""
        if not self.ok:
            return DISABLED_IMU_READING
        with self._lock:
            if self._history:
                return self._history[-1]
        return DISABLED_IMU_READING

    def swing_at(self, timestamp: float | None, max_age: float = 0.25) -> float | None:
        """Swing at the instant `timestamp`, or None if no sample is close.

        The DOA angle and the swing correction have to describe the *same*
        moment. Adding a swing sampled 300 ms after the DOA reading would inject
        exactly the oscillation this module exists to remove, so a stale or
        missing match returns None and the caller publishes the uncorrected
        angle rather than a confidently wrong one.
        """
        if not self.ok or timestamp is None:
            return None
        best: IMUReading | None = None
        best_gap = float("inf")
        with self._lock:
            for reading in reversed(self._history):
                gap = abs(reading.timestamp - timestamp)
                if gap < best_gap:
                    best_gap = gap
                    best = reading
                elif reading.timestamp < timestamp:
                    # Scanning newest-first, the gap shrinks then grows; once it
                    # grows on a sample older than the target we are past it.
                    break
        if best is None or best_gap > max_age:
            return None
        return best.swing

    def debug_snapshot(self) -> str:
        """One-line human-readable state, for the bring-up check in the README."""
        if not self.ok:
            return f"IMU off ({self.status})"
        with self._lock:
            accel = self._accel
            yaw, ref, swing = self._yaw, self._ref, self._swing
            rate, mean, turning = self._yaw_rate, self._gyro_mean, self._turning
            depth = len(self._history)
        return (
            f"yaw={yaw:7.2f} ref={ref:7.2f} swing={swing:+7.2f} "
            f"rate={rate:+7.2f} mean0.5s={mean:+7.2f} "
            f"state={'TURN' if turning else 'still'} "
            f"acc=({accel[0]:+.2f},{accel[1]:+.2f},{accel[2]:+.2f}) "
            f"buf={depth}"
        )

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None
        self._close_bus()

    def _close_bus(self) -> None:
        if self._bus is not None:
            try:
                self._bus.close()
            except Exception:
                pass
            self._bus = None


def _subtract(
    left: tuple[float, float, float], right: tuple[float, float, float]
) -> tuple[float, float, float]:
    return (left[0] - right[0], left[1] - right[1], left[2] - right[2])


def _norm(vector: tuple[float, float, float]) -> float:
    return math.sqrt(vector[0] ** 2 + vector[1] ** 2 + vector[2] ** 2)


def _normalize(
    vector: tuple[float, float, float]
) -> tuple[float, float, float] | None:
    magnitude = _norm(vector)
    if magnitude < 1e-6:
        return None
    return (vector[0] / magnitude, vector[1] / magnitude, vector[2] / magnitude)
