"""ReSpeaker USB DSP direction-of-arrival reader.

The ReSpeaker 4 Mic Array computes DOA on-device and exposes it over a USB
vendor control transfer (`DOAANGLE`, read-only, 0..359). That path is
independent of the audio stream, so it works with the 6-channel firmware and
needs no alignment with the inference window: we poll it in the background and
attach the freshest reading when a detection is published.

Audio-domain GCC-PHAT estimation is intentionally NOT implemented here — the
DSP already does it.
"""
from __future__ import annotations

import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

# Measured with `lsusb` on this unit: 2886:0018 Seeed ReSpeaker 4 Mic Array.
RESPEAKER_USB_VENDOR_ID = 0x2886
RESPEAKER_USB_PRODUCT_ID = 0x0018

TUNING_DIR = Path(__file__).resolve().parent / "usb_4_mic_array"


@dataclass(frozen=True)
class DOAReading:
    """A DOA snapshot. `raw_angle is None` means "no direction available".

    `timestamp` is the `time.monotonic()` instant `raw_angle` was read off the
    DSP, not the instant `snapshot()` was called. The IMU de-swing correction
    has to be sampled at the same moment as the angle it corrects, so this has
    to survive the trip through the publish path.
    """

    raw_angle: int | None
    source: str
    status: str
    timestamp: float | None = None


DISABLED_READING = DOAReading(None, "none", "disabled", None)


class DOAReader:
    """Background poller for the ReSpeaker USB DSP angle."""

    def __init__(
        self,
        *,
        enabled: bool = True,
        poll_interval: float = 0.1,
        vendor_id: int = RESPEAKER_USB_VENDOR_ID,
        product_id: int = RESPEAKER_USB_PRODUCT_ID,
    ) -> None:
        self.ok = False
        self.tuning = None
        self.status = "disabled"
        self.poll_interval = max(0.02, float(poll_interval))
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._last_angle: int | None = None
        self._last_voice: bool | None = None
        self._last_read_at: float | None = None
        # `_last_read_at` tracks liveness (did the poll thread run?) while
        # `_last_angle_at` tracks the age of `_last_angle` specifically: a poll
        # that comes back with no angle refreshes the former but not the latter.
        self._last_angle_at: float | None = None
        self._last_error: str | None = None

        if not enabled:
            print("[DOA] --disable-doa: 방향 추정을 끕니다.", file=sys.stderr, flush=True)
            return

        try:
            import usb.core  # type: ignore

            if str(TUNING_DIR) not in sys.path:
                sys.path.insert(0, str(TUNING_DIR))
            from tuning import Tuning  # type: ignore

            device = usb.core.find(idVendor=vendor_id, idProduct=product_id)
            if device is None:
                self.status = "usb_device_not_found"
                print(
                    f"[DOA] ReSpeaker USB 제어 장치를 찾지 못했습니다 "
                    f"({vendor_id:04x}:{product_id:04x}). lsusb로 VID:PID를 확인하세요.",
                    file=sys.stderr,
                    flush=True,
                )
                return

            self.tuning = Tuning(device)
            # 권한/펌웨어 문제는 첫 읽기에서 바로 드러난다.
            angle = self.tuning.direction
            if angle is None:
                self.status = "usb_no_angle"
                print("[DOA] DOAANGLE을 읽지 못했습니다.", file=sys.stderr, flush=True)
                return

            self.ok = True
            self.status = "enabled"
            self._last_angle = int(float(angle)) % 360
            self._last_read_at = self._last_angle_at = time.monotonic()
            self._thread = threading.Thread(
                target=self._poll_loop,
                name="doa-usb-poll",
                daemon=True,
            )
            self._thread.start()
            print(
                f"[DOA] ReSpeaker USB DOA 활성화 (첫 각도 {self._last_angle}도, "
                f"poll={self.poll_interval:.2f}s)",
                file=sys.stderr,
                flush=True,
            )
        except Exception as exc:
            self.status = "usb_unavailable"
            self._last_error = type(exc).__name__
            hint = ""
            if "Access denied" in str(exc) or "permission" in str(exc).lower():
                hint = (
                    "\n[DOA] udev 룰이 필요합니다: /etc/udev/rules.d/60-respeaker.rules 에\n"
                    '      SUBSYSTEM=="usb", ATTRS{idVendor}=="2886", '
                    'ATTRS{idProduct}=="0018", MODE="0666"'
                )
            print(f"[DOA] 비활성화: {exc!r}{hint}", file=sys.stderr, flush=True)

    def describe(self) -> str:
        return self.status

    def _read_device(self) -> tuple[int | None, bool | None]:
        if self.tuning is None:
            return None, None

        voice: bool | None
        try:
            voice_value = self.tuning.is_voice()
            voice = None if voice_value is None else bool(int(voice_value))
        except Exception:
            voice = None

        angle = self.tuning.direction
        if angle is None:
            return None, voice
        return int(float(angle)) % 360, voice

    def _poll_once(self) -> None:
        if not self.ok or self.tuning is None:
            return
        try:
            with self._lock:
                angle, voice = self._read_device()
                # Stamped next to the transfer, not at snapshot() time: this is
                # the instant the IMU swing has to be sampled at.
                read_at = time.monotonic()
            self._last_read_at = read_at
            self._last_voice = voice
            if angle is not None:
                self._last_angle = angle
                self._last_angle_at = read_at
            self.status = "enabled"
            self._last_error = None
        except Exception as exc:
            self.status = "usb_read_error"
            self._last_error = type(exc).__name__

    def _poll_loop(self) -> None:
        while not self._stop_event.wait(self.poll_interval):
            self._poll_once()

    def snapshot(self) -> DOAReading:
        """Latest angle, or a reading with `raw_angle=None` and a reason.

        Never returns 0 as a stand-in for "unknown" — the app would render that
        as a real direction.
        """
        if not self.ok:
            status = self.status
            if self._last_error:
                status = f"{status}:{self._last_error}"
            return DOAReading(None, "none", status, None)

        angle = self._last_angle
        read_at = self._last_angle_at
        if angle is None:
            status = self._last_error and f"usb_no_angle:{self._last_error}"
            return DOAReading(None, "usb", status or "usb_no_angle", None)

        age = None if read_at is None else time.monotonic() - read_at
        if age is not None and age > max(1.0, self.poll_interval * 5.0):
            return DOAReading(None, "usb", "usb_stale", read_at)

        status = "usb_no_voice" if self._last_voice is False else "usb_active"
        return DOAReading(angle, "usb", status, read_at)

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None
        if self.tuning is not None:
            try:
                self.tuning.close()
            except Exception:
                pass
            self.tuning = None
