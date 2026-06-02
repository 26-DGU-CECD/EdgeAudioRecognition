from __future__ import annotations


def angle_to_cardinal(angle: float) -> str:
    corrected = float(angle) % 360.0
    if corrected < 45.0 or corrected >= 315.0:
        return "북"
    if corrected < 135.0:
        return "동"
    if corrected < 225.0:
        return "남"
    return "서"


def corrected_angle(raw_angle: float, north_offset: float) -> int:
    return int(round((float(raw_angle) - float(north_offset)) % 360.0)) % 360
