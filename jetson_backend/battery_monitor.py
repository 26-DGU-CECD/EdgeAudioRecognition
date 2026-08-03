"""UPS battery telemetry (X1200/X1201/X1202) for the BLE payload."""
from __future__ import annotations

import struct
import sys
import time

GAUGE_ADDRESS = 0x36
VOLTAGE_REGISTER = 0x02
CAPACITY_REGISTER = 0x04
LOW_BATTERY_PERCENT = 20.0

UNAVAILABLE = {
    "battery_percent": None,
    "battery_voltage": None,
    "battery_low": None,
}


class BatteryMonitor:
    """Reads the UPS fuel gauge over I2C and caches it for the publish path.

    A window can be published several times per second, so the gauge is polled at
    most once per ``min_interval_seconds`` and every payload reuses that reading.
    The first read happens in the constructor and is allowed to fail loudly, so
    main can disable battery reporting when no UPS is on the bus. Later failures
    only blank the fields, because losing I2C must not stop the audio stream.
    """

    def __init__(self, bus_number: int = 1, min_interval_seconds: float = 5.0) -> None:
        import smbus2

        self.bus_number = int(bus_number)
        self.min_interval_seconds = max(0.0, float(min_interval_seconds))
        self._bus = smbus2.SMBus(self.bus_number)
        self._cache = dict(UNAVAILABLE)
        self._read_at: float | None = None
        self._warned = False
        self._refresh(time.monotonic(), strict=True)

    def snapshot(self) -> dict:
        now = time.monotonic()
        if self._read_at is None or now - self._read_at >= self.min_interval_seconds:
            self._refresh(now)
        return dict(self._cache)

    def describe(self) -> str:
        state = self.snapshot()
        if state["battery_percent"] is None:
            return "unavailable"
        return f"{state['battery_percent']:.1f}% ({state['battery_voltage']:.3f}V)"

    def close(self) -> None:
        try:
            self._bus.close()
        except Exception:
            pass

    def _refresh(self, now: float, strict: bool = False) -> None:
        try:
            voltage = self._read_word(VOLTAGE_REGISTER) * 1.25 / 1000 / 16
            percent = self._read_word(CAPACITY_REGISTER) / 256
        except OSError as exc:
            if strict:
                raise
            if not self._warned:
                print(
                    f"배터리 읽기 실패(I2C): {exc} | 배터리 값 없이 계속 전송합니다.",
                    file=sys.stderr,
                    flush=True,
                )
                self._warned = True
            self._cache = dict(UNAVAILABLE)
            self._read_at = now
            return

        self._warned = False
        percent = min(100.0, max(0.0, percent))
        self._cache = {
            "battery_percent": round(percent, 1),
            "battery_voltage": round(voltage, 3),
            "battery_low": percent < LOW_BATTERY_PERCENT,
        }
        self._read_at = now

    def _read_word(self, register: int) -> int:
        raw = self._bus.read_word_data(GAUGE_ADDRESS, register)
        return struct.unpack("<H", struct.pack(">H", raw))[0]
