"""Detector-level label gate: engine skipping, partial DecisionGate, BLE parsing.

Fake detectors only — no torch/TF. Run from jetson_backend/: python -m pytest tests
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import runtime_config  # noqa: E402
from ble_result_builder import (  # noqa: E402
    build_control_state,
    from_wire_label,
    parse_disabled_labels,
    wire_label,
)
from decision import DecisionGate  # noqa: E402
from parallel_inference import ParallelInferenceEngine  # noqa: E402

CLASSES = {
    "yamnet": ["siren", "scream", "crying", "car_horn"],
    "efficientat": ["bicycle_bell", "baby_cry", "gunshot", "glass_break"],
    "mobilenetv4": ["water", "knock", "dog", "cat"],
}
# siren이 1등(margin .4), gunshot이 2등(margin .2), dog도 threshold를 넘음(margin .1).
PROBS = {
    "siren": 0.9, "scream": 0.1, "crying": 0.2, "car_horn": 0.3,
    "bicycle_bell": 0.05, "baby_cry": 0.1, "gunshot": 0.7, "glass_break": 0.2,
    "water": 0.1, "knock": 0.2, "dog": 0.6, "cat": 0.3,
}
THRESHOLDS = {
    name: 0.5 for classes in CLASSES.values() for name in classes
}


class FakeDetector:
    def __init__(self, classes: list[str]) -> None:
        self.target_classes = list(classes)
        self.calls = 0

    def predict_proba_batch(self, waveforms_16k: np.ndarray) -> np.ndarray:
        self.calls += 1
        row = [PROBS[name] for name in self.target_classes]
        return np.asarray([row] * len(waveforms_16k), dtype=np.float32)


def make_engine(concurrent: bool = False) -> ParallelInferenceEngine:
    detectors = {name: FakeDetector(classes) for name, classes in CLASSES.items()}
    return ParallelInferenceEngine(detectors, concurrent=concurrent)


def make_gate(engine: ParallelInferenceEngine) -> DecisionGate:
    return DecisionGate(THRESHOLDS, classes=engine.classes, debounce_seconds=3.0)


WINDOW = np.zeros(runtime_config.WINDOW_SAMPLES, dtype=np.float32)


@pytest.mark.parametrize("concurrent", [False, True])
def test_fully_disabled_detector_is_not_called(concurrent: bool) -> None:
    engine = make_engine(concurrent)
    engine.set_disabled_labels(CLASSES["yamnet"])

    result = engine.predict(WINDOW)

    assert engine.detectors["yamnet"].calls == 0
    assert engine.detectors["efficientat"].calls == 1
    assert engine.detectors["mobilenetv4"].calls == 1
    assert result.skipped_detectors == ("yamnet",)
    assert "yamnet" not in result.detector_latency_ms
    assert set(result.probabilities) == set(CLASSES["efficientat"] + CLASSES["mobilenetv4"])
    engine.close()


@pytest.mark.parametrize("concurrent", [False, True])
def test_partially_disabled_detector_runs_and_drops_only_that_label(concurrent: bool) -> None:
    engine = make_engine(concurrent)
    full = engine.predict(WINDOW).probabilities

    engine.set_disabled_labels(["dog"])
    result = engine.predict(WINDOW)

    assert engine.detectors["mobilenetv4"].calls == 2
    assert result.skipped_detectors == ()
    assert "dog" not in result.probabilities
    assert result.probabilities == {k: v for k, v in full.items() if k != "dog"}
    engine.close()


def test_all_disabled() -> None:
    engine = make_engine()
    gate = make_gate(engine)
    engine.set_disabled_labels(engine.classes)

    result = engine.predict(WINDOW)
    decision = gate.evaluate(result.probabilities, now=0.0)

    assert engine.all_disabled
    assert engine.active_detectors == ()
    assert result.probabilities == {}
    assert all(d.calls == 0 for d in engine.detectors.values())
    assert decision.best_label is None
    assert decision.nearest_label is None
    assert decision.candidates == []


def test_decision_gate_partial_keys() -> None:
    gate = make_gate(make_engine())
    partial = {name: PROBS[name] for name in CLASSES["mobilenetv4"] + ["gunshot"]}

    decision = gate.evaluate(partial, now=0.0)

    assert decision.candidates == ["gunshot", "dog"]  # margin .2 > .1
    assert decision.best_label == "gunshot"
    assert decision.best_margin == pytest.approx(0.2)


def test_disabling_winner_promotes_runner_up() -> None:
    engine = make_engine()
    gate = make_gate(engine)

    before = gate.evaluate(engine.predict(WINDOW).probabilities, now=0.0)
    engine.set_disabled_labels(CLASSES["yamnet"])
    after = gate.evaluate(engine.predict(WINDOW).probabilities, now=1.0)

    assert before.best_label == "siren"
    assert before.candidates[1] == "gunshot"
    assert after.best_label == "gunshot"


def test_unknown_labels_are_ignored() -> None:
    engine = make_engine()

    applied = engine.set_disabled_labels(["dog", "unicorn", ""])

    assert applied == frozenset({"dog"})
    assert engine.disabled_labels == frozenset({"dog"})
    assert len(engine.active_detectors) == 3


def test_reset_labels_clears_debounce() -> None:
    gate = make_gate(make_engine())
    gate.evaluate(PROBS, now=0.0)
    assert gate.evaluate(PROBS, now=1.0).new_events == []

    gate.reset_labels(["siren"])

    assert gate.evaluate(PROBS, now=2.0).new_events == ["siren"]


def test_wire_label_round_trip() -> None:
    assert wire_label("bicycle_bell") == "bicycle"
    assert from_wire_label("bicycle") == "bicycle_bell"
    assert from_wire_label("dog") == "dog"


def test_parse_disabled_labels_converts_wire_names() -> None:
    raw = '{"disabled_labels": ["bicycle", "dog", " "]}'.encode("utf-8")
    assert parse_disabled_labels(raw) == ["bicycle_bell", "dog"]


@pytest.mark.parametrize(
    "raw",
    [
        b"{not json",
        b"\xff\xfe",
        b"[1, 2]",
        b'{"disabled_labels": "dog"}',
        b'{"disabled_labels": ["dog", 3]}',
        b"{}",
    ],
)
def test_parse_disabled_labels_rejects_bad_input(raw: bytes) -> None:
    with pytest.raises(ValueError):
        parse_disabled_labels(raw)


def test_control_state_uses_wire_names() -> None:
    assert build_control_state({"bicycle_bell", "dog"}) == {
        "disabled_labels": ["bicycle", "dog"]
    }


# --------------------------------------------------------------------------- #
# Controller: idle packets instead of silence
# --------------------------------------------------------------------------- #
class Recorder:
    def __init__(self) -> None:
        self.packets: list[dict] = []

    def publish(self, data: dict) -> None:
        self.packets.append(data)


def make_controller(engine: ParallelInferenceEngine):
    from audio_buffer import SlidingWindowBuffer
    from audio_level_meter import AudioLevelMeter
    from audio_queue import AudioQueue
    from audio_stream_controller import AudioStreamController
    from db_threshold_gate import DbThresholdGate

    recorder = Recorder()
    controller = AudioStreamController(
        audio_queue=AudioQueue(1.0, runtime_config.SAMPLE_RATE),
        window_buffer=SlidingWindowBuffer(runtime_config.WINDOW_SAMPLES, 16000),
        level_meter=AudioLevelMeter(),
        threshold_gate=DbThresholdGate(45.0),
        inference_engine=engine,
        decision_gate=make_gate(engine),
        publisher=recorder,
        skip_low_db=False,
    )
    return controller, recorder


LOUD = np.full(runtime_config.WINDOW_SAMPLES, 0.1, dtype=np.float32)


def test_controller_all_disabled_sends_idle_without_inference() -> None:
    engine = make_engine()
    controller, recorder = make_controller(engine)
    controller.set_disabled_labels(engine.classes)

    controller.process_window(LOUD)

    assert recorder.packets[-1]["status"] == "idle"
    assert recorder.packets[-1]["reason"] == "all_disabled"
    assert recorder.packets[-1]["db"] > 0
    assert all(d.calls == 0 for d in engine.detectors.values())


def test_controller_error_sends_idle() -> None:
    engine = make_engine()

    def boom(_waveforms):  # noqa: ANN001
        raise RuntimeError("boom")

    engine.detectors["efficientat"].predict_proba_batch = boom
    controller, recorder = make_controller(engine)

    controller.process_window(LOUD)

    assert recorder.packets[-1]["reason"] == "error"


def test_controller_partial_gate_reports_remaining_labels() -> None:
    engine = make_engine()
    controller, recorder = make_controller(engine)
    controller.set_disabled_labels(CLASSES["yamnet"])

    controller.process_window(LOUD)

    assert recorder.packets[-1]["status"] == "ok"
    assert recorder.packets[-1]["label"] == "gunshot"


# --------------------------------------------------------------------------- #
# ControlCharacteristic with dbus stubbed out (dbus/gi are Pi-only)
# --------------------------------------------------------------------------- #
@pytest.fixture
def gatt_objects(monkeypatch):
    import types

    dbus = types.ModuleType("dbus")
    dbus.service = types.ModuleType("dbus.service")
    dbus.exceptions = types.ModuleType("dbus.exceptions")
    dbus.exceptions.DBusException = type("DBusException", (Exception,), {})
    dbus.service.Object = type("Object", (), {"__init__": lambda self, bus, path: None})
    dbus.service.method = lambda *a, **k: (lambda fn: fn)
    dbus.service.signal = lambda *a, **k: (lambda fn: fn)
    dbus.Array = lambda items, signature=None: list(items)
    dbus.Byte = int
    dbus.ObjectPath = str
    dbus.Boolean = bool
    dbus.String = str
    dbus.Dictionary = lambda d, signature=None: dict(d)
    dbus.SystemBus = object
    gi = types.ModuleType("gi")
    gi.repository = types.ModuleType("gi.repository")
    gi.repository.GLib = types.SimpleNamespace(idle_add=lambda fn: fn())
    for name, module in {
        "dbus": dbus,
        "dbus.service": dbus.service,
        "dbus.exceptions": dbus.exceptions,
        "gi": gi,
        "gi.repository": gi.repository,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)
    for name in ("ble_gatt_objects", "ble_utils"):
        monkeypatch.delitem(sys.modules, name, raising=False)
    import ble_gatt_objects

    return ble_gatt_objects


def test_control_write_never_raises(gatt_objects) -> None:
    import json

    engine = make_engine()
    service = gatt_objects.Service(None, 0, "uuid")
    control = gatt_objects.ControlCharacteristic(
        None, 1, service, engine.set_disabled_labels, lambda: engine.disabled_labels
    )

    for raw in (b"{broken", b'{"disabled_labels": 5}', b'{"disabled_labels": ["unicorn"]}'):
        control.WriteValue(list(raw), {})
    assert engine.disabled_labels == frozenset()

    control.WriteValue(list(b'{"disabled_labels": ["bicycle", "baby_cry", "gunshot", "glass_break"]}'), {})
    assert engine.active_detectors == ("yamnet", "mobilenetv4")
    assert json.loads(bytes(control.ReadValue({}))) == {
        "disabled_labels": ["baby_cry", "bicycle", "glass_break", "gunshot"]
    }

    def explode(_labels):  # noqa: ANN001
        raise RuntimeError("callback failure")

    control.on_disabled_labels = explode
    control.WriteValue(list(b'{"disabled_labels": []}'), {})  # must not raise
