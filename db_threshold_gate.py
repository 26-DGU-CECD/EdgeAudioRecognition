from __future__ import annotations

from realtime_inference import db_gate_threshold


class DbThresholdGate:
    def __init__(self, min_db: float) -> None:
        self.min_db = float(min_db)
        self.min_dbfs = float(db_gate_threshold(min_db))

    def is_over_threshold(self, dbfs: float) -> bool:
        return float(dbfs) >= self.min_dbfs

    def reason(self, dbfs: float) -> str:
        return f"low_signal {dbfs:+.1f}<{self.min_dbfs:+.1f}dBFS"
