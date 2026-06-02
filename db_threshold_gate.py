from __future__ import annotations

from realtime_inference import db_gate_threshold


class DbThresholdGate:
    """음압이 설정 임계값 이상인지 판단."""

    def __init__(self, min_db: float) -> None:
        self.min_db = float(min_db)
        self.min_dbfs = float(db_gate_threshold(min_db))

    def convert_threshold(self, min_db: float) -> float:
        return float(db_gate_threshold(min_db))

    def is_over_threshold(self, dbfs: float) -> bool:
        return float(dbfs) >= self.min_dbfs

    def low_signal_reason(self, dbfs: float) -> str:
        return f"low_signal {dbfs:+.1f}<{self.min_dbfs:+.1f}dBFS"
