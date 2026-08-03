"""Build the JSON packets consumed by the Flutter app (EdgeApp).

Contract source of truth: `lib/models/sound_packet.dart` (`SoundPacket.fromJson`)
and `lib/ble/ble_sound_service.dart` (`soundPacketFromBleJson`).

Key app rules this module has to satisfy:
- `status` must be exactly "ok" or the app drops the packet entirely.
- `db` is a POSITIVE dB value, not dBFS: `db = max(0, dBFS + db_offset)`.
- Home screen shows a packet only when `status=="ok" && score>=0.7 && db>=45.0`.
- `angle` is null (never 0) when DOA is unavailable; the app treats a present
  angle as a valid direction.
- The `direction` cardinal field was removed on both sides; only
  `direction_text` remains.
"""
from __future__ import annotations

from typing import Dict, Iterable

from decision import Decision

SOURCE = "jetson_backend_main"

# 앱이 사용하는 위험도 등급. main.py의 11개 클래스 기준으로 다시 작성했다.
# (참고 코드 realtime_inference_ble_doa.py의 DANGER/CAUTION_LABELS는
#  alarm_siren/horn/glass_shatter 같은 다른 라벨 체계라 그대로 쓸 수 없다.)
DANGER_LABELS = {
    "gunshot",
    "siren",
    "scream",
    "glass_break",
}

CAUTION_LABELS = {
    "baby_cry",
    "crying",
    "knock",
    "bicycle_bell",
    "water",
}

# 나머지(dog, cat)는 "info".

# 백엔드 클래스명 -> 앱 `_soundLabelKoMap` 키.
# 앱에는 'bicycle_bell' 키가 없고 'bicycle'이 있으므로 전송 시점에 바꿔준다.
# 'knock'은 앱 쪽 매핑표에 한 줄 추가했다(작업 4).
WIRE_LABELS = {
    "bicycle_bell": "bicycle",
}

# `SoundPacket.directionLabel` / `home_page.dart`의 CompassView와 같은 규칙.
# angle 0 = 착용자 뒤쪽, 90 = 왼쪽, 180 = 앞쪽, 270 = 오른쪽 (시계방향).
RELATIVE_DIRECTIONS = ("뒤쪽", "왼쪽", "앞쪽", "오른쪽")


def wire_label(label: str) -> str:
    """Rename backend classes that the app's Korean label map does not know."""
    return WIRE_LABELS.get(str(label), str(label))


def risk_level(label: str) -> str:
    key = str(label).strip().lower()
    if key in DANGER_LABELS:
        return "danger"
    if key in CAUTION_LABELS:
        return "caution"
    return "info"


def app_db_from_dbfs(dbfs: float, offset: float) -> float:
    """dBFS -> the positive dB scale the app displays and thresholds on."""
    return round(max(0.0, float(dbfs) + float(offset)), 1)


def corrected_angle(raw_angle: float, north_offset: float) -> int:
    return int(round((float(raw_angle) - float(north_offset)) % 360.0)) % 360


def relative_direction(angle: float) -> str:
    """Same mapping as `SoundPacket.directionLabel` in the app."""
    normalized = (float(angle) % 360.0 + 360.0) % 360.0
    return RELATIVE_DIRECTIONS[int((normalized + 45.0) % 360.0) // 90]


def build_app_packet(
    *,
    timestamp: str,
    label: str,
    display_label: str | None = None,
    score: float,
    infer_sec: float,
    total_sec: float,
    chunk_dbfs: float,
    db_offset: float,
    raw_line: str,
    probabilities: Dict[str, float] | None = None,
    raw_angle: int | None = None,
    north_offset: float = 0.0,
    doa_status: str = "disabled",
    full_packet: bool = False,
) -> dict:
    """Build a `status: "ok"` sound packet."""
    sent_label = wire_label(label)

    if raw_angle is None:
        angle: float | None = None
        angle_raw: float | None = None
        direction_text = ""
    else:
        corrected = corrected_angle(raw_angle, north_offset)
        angle = float(corrected)
        angle_raw = float(raw_angle)
        direction_text = f"{relative_direction(corrected)} {corrected}도"

    packet = {
        "status": "ok",
        "time": timestamp,
        "label": sent_label,
        "display_label": display_label or sent_label,
        "score": round(float(score), 4),
        "infer_sec": round(float(infer_sec), 3),
        "total_sec": round(float(total_sec), 3),
        "db": app_db_from_dbfs(chunk_dbfs, db_offset),
        "level": risk_level(label),
        "angle": angle,
        "angle_raw": angle_raw,
        "has_doa": raw_angle is not None,
        "direction_text": direction_text,
        "doa_status": doa_status,
        "raw": raw_line,
    }

    if full_packet:
        packet["source"] = SOURCE
        packet["dbfs"] = round(float(chunk_dbfs), 2)
        packet["items"] = _build_items(probabilities or {})

    return packet


def build_idle_packet(
    *,
    timestamp: str,
    reason: str,
    chunk_dbfs: float,
    db_offset: float,
) -> dict:
    """Non-detection heartbeat.

    `status` is deliberately NOT "ok": the app drops it as a sound packet
    (`soundPacketFromBleJson`) but `_handleJson` still reads `battery_percent`
    out of it first, so battery keeps flowing while nothing is detected.
    """
    return {
        "status": "idle",
        "time": timestamp,
        "reason": reason,
        "db": app_db_from_dbfs(chunk_dbfs, db_offset),
    }


def _build_items(probabilities: Dict[str, float]) -> list[dict]:
    ranked: Iterable[tuple[str, float]] = sorted(
        probabilities.items(),
        key=lambda item: -float(item[1]),
    )
    items = []
    for label, score in ranked:
        sent_label = wire_label(label)
        items.append(
            {
                "label": sent_label,
                "display_label": sent_label,
                "score": round(float(score), 4),
            }
        )
    return items


def summarize_decision(decision: Decision) -> tuple[str, float]:
    """Best detected class, or the nearest miss when nothing crossed."""
    if decision.best_label:
        return decision.best_label, float(decision.best_probability)
    return decision.nearest_label, float(decision.nearest_probability)
