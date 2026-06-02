from __future__ import annotations

from typing import Dict

from angle_utils import angle_to_cardinal, corrected_angle
from constants import CARDINAL_SUFFIX, CAUTION_LABELS, DANGER_LABELS


def risk_level(label: str) -> str:
    key = str(label).strip().lower()
    if key in DANGER_LABELS:
        return "danger"
    if key in CAUTION_LABELS:
        return "caution"
    return "info"


def app_db_from_dbfs(dbfs: float, offset: float) -> float:
    return round(max(0.0, float(dbfs) + float(offset)), 1)


class AppSoundPacketBuilder:
    """추론 결과, dB, DOA 정보를 앱 전송용 JSON dict로 구성."""

    def __init__(self, *, north_offset: float, db_offset: float, full_packet: bool) -> None:
        self.north_offset = float(north_offset)
        self.db_offset = float(db_offset)
        self.full_packet = bool(full_packet)

    def build(
        self,
        *,
        timestamp: str,
        label: str,
        score: float,
        scores: Dict[str, float],
        infer_sec: float,
        total_sec: float,
        chunk_dbfs: float,
        status_text: str,
        raw_line: str,
        raw_angle: int | None,
        doa_status: str,
        doa_source: str,
    ) -> dict:
        if raw_angle is None:
            angle = None
            direction = ""
            direction_text = ""
        else:
            angle = corrected_angle(raw_angle, self.north_offset)
            direction = angle_to_cardinal(angle)
            direction_text = f"{CARDINAL_SUFFIX[direction]} {angle}도"

        app_db = app_db_from_dbfs(chunk_dbfs, self.db_offset)
        packet = {
            "status": "ok",
            "time": timestamp,
            "label": label,
            "score": round(float(score), 6),
            "infer_sec": round(float(infer_sec), 3),
            "total_sec": round(float(total_sec), 3),
            "db": app_db,
            "level": risk_level(label),
            "direction": direction,
            "angle": float(angle) if angle is not None else None,
            "angle_raw": float(raw_angle) if raw_angle is not None else None,
            "direction_text": direction_text,
            "doa_status": doa_status,
            "doa_source": doa_source,
            "has_doa": raw_angle is not None,
        }

        if self.full_packet:
            packet["display_label"] = label
            packet["dbfs"] = round(float(chunk_dbfs), 2)
            packet["status_text"] = status_text
            packet["raw"] = raw_line
            packet["items"] = [
                {
                    "label": item_label,
                    "display_label": item_label,
                    "score": round(float(item_score), 6),
                    "direction": direction,
                }
                for item_label, item_score in scores.items()
            ]

        return packet
