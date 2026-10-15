from __future__ import annotations

import json
from pathlib import Path

import config


RESULTS = {
    "A EfficientAT": config.EFFICIENTAT_OUTPUT,
    "B YAMNet": config.YAMNET_OUTPUT,
    "C MobileNetV4": config.MOBILENETV4_OUTPUT,
    "D Full ensemble": config.FULL_ENSEMBLE_OUTPUT,
    "E EfficientAT specialist": config.EFFICIENTAT_SPECIALIST_OUTPUT,
    "E YAMNet specialist": config.YAMNET_SPECIALIST_OUTPUT,
    "E MobileNetV4 specialist": config.MOBILENETV4_SPECIALIST_OUTPUT,
    "F Specialist weighted ensemble": config.SPECIALIST_ENSEMBLE_OUTPUT,
}


for name, directory in RESULTS.items():
    path = Path(directory) / "test_metrics.json"
    if not path.exists():
        print(f"{name:34s} 결과 없음")
        continue
    metrics = json.loads(path.read_text(encoding="utf-8"))
    print(
        f"{name:34s} "
        f"accuracy={metrics['accuracy']:.4f}, "
        f"macro_f1={metrics['macro_f1']:.4f}"
    )
