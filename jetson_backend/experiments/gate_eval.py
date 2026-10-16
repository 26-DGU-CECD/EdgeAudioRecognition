#!/usr/bin/env python3
"""Detector-level label gate experiments.

  scores  : A (remaining scores unchanged) + B (masked runner-ups surfacing)
  cost    : C (per-detector load/latency/RSS + main.py --input-wav under 5 gates)

Run from jetson_backend/:
  python experiments/gate_eval.py scores --data-root <test split dir> --out <dir>
  python experiments/gate_eval.py cost   --data-root <test split dir> --out <dir>
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

BACKEND_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_DIR))

import runtime_config  # noqa: E402

# 평가 폴더 -> 그 클래스를 담당하는 detector. background는 세 detector 각각을 끈다.
FOLDER_OWNER = {
    "siren": "yamnet",
    "fire_siren": "yamnet",
    "gas_siren": "yamnet",
    "crying": "yamnet",
    "scream": "yamnet",
    "gunshot": "efficientat",
}
DETECTORS = ("efficientat", "yamnet", "mobilenetv4")


def load_windows(data_root: Path, limit: int | None) -> tuple[list[str], list[str], np.ndarray]:
    """Every clip -> 2 s windows (hop 1 s, zero-padded tail). Returns folder, path, windows."""
    import soundfile as sf

    win = runtime_config.WINDOW_SAMPLES
    hop = runtime_config.SAMPLE_RATE
    folders, paths, windows = [], [], []
    for folder in sorted(p for p in data_root.iterdir() if p.is_dir()):
        files = sorted(folder.glob("*.wav"))[:limit]
        for path in files:
            wav, sr = sf.read(path, dtype="float32", always_2d=False)
            if wav.ndim > 1:
                wav = wav.mean(axis=1)
            if sr != runtime_config.SAMPLE_RATE:
                raise SystemExit(f"{path}: sr={sr}, 16 kHz만 지원")
            if len(wav) < win:
                wav = np.pad(wav, (0, win - len(wav)))
            for start in range(0, len(wav) - win + 1, hop):
                folders.append(folder.name)
                paths.append(str(path))
                windows.append(wav[start:start + win])
    return folders, paths, np.stack(windows)


def run_config(engine, gate, windows: np.ndarray, disabled: list[str]) -> dict:
    """All windows under one gate. Probabilities of disabled/skipped labels are NaN."""
    engine.set_disabled_labels(disabled)
    gate.reset_labels(engine.classes)
    probs = np.full((len(windows), len(engine.classes)), np.nan, dtype=np.float64)
    best, candidates, latency = [], [], []
    for i, window in enumerate(windows):
        result = engine.predict(window)
        decision = gate.evaluate(result.probabilities, now=float(i))
        for j, name in enumerate(engine.classes):
            if name in result.probabilities:
                probs[i, j] = result.probabilities[name]
        best.append(decision.best_label)
        candidates.append(decision.candidates)
        latency.append(result.total_latency_ms)
        if i % 250 == 0:
            print(f"  {i}/{len(windows)}", flush=True)
    return {"probs": probs, "best": best, "candidates": candidates, "latency": latency}


def dist(labels) -> str:
    counts = Counter("(none)" if x is None else x for x in labels)
    return ", ".join(f"{k} {v}" for k, v in counts.most_common())


def cmd_scores(args: argparse.Namespace) -> int:
    from decision import DecisionGate, load_thresholds
    from detector_registry import build_detectors
    from parallel_inference import ParallelInferenceEngine

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    folders, paths, windows = load_windows(Path(args.data_root), args.limit_per_class)
    print(f"windows={len(windows)} folders={dict(Counter(folders))}", flush=True)

    engine = ParallelInferenceEngine(build_detectors(None, batch_size=1, torch_threads=args.torch_threads))
    gate = DecisionGate(
        load_thresholds(runtime_config.THRESHOLDS_JSON), classes=engine.classes
    )
    owned = {name: list(det.target_classes) for name, det in engine.detectors.items()}

    configs = {"all_on": []} | {f"{name}_off": owned[name] for name in DETECTORS}
    runs = {}
    for key, disabled in configs.items():
        print(f"[run] {key} disabled={disabled}", flush=True)
        started = time.perf_counter()
        runs[key] = run_config(engine, gate, windows, disabled)
        print(f"[run] {key} done in {time.perf_counter() - started:.0f}s", flush=True)
    # 결정성 대조군: 같은 설정을 다시 돌렸을 때도 비트 단위로 같은지.
    n_ctrl = min(args.control_windows, len(windows))
    control = run_config(engine, gate, windows[:n_ctrl], [])

    report: list[str] = []
    classes = engine.classes

    # ---- A ----
    report.append("## A. 남은 라벨 점수 불변\n")
    report.append("| 설정 | 비교 라벨 수 | 윈도우 | np.array_equal | 최대 절대차 |")
    report.append("|---|---|---|---|---|")
    ctrl_equal = np.array_equal(runs["all_on"]["probs"][:n_ctrl], control["probs"])
    report.append(f"| all_on 재실행(대조군, {n_ctrl}개) | 12 | {n_ctrl} | {ctrl_equal} | "
                  f"{np.nanmax(np.abs(runs['all_on']['probs'][:n_ctrl] - control['probs'])):.3g} |")
    a_results = {}
    for name in DETECTORS:
        keep = [j for j, c in enumerate(classes) if c not in owned[name]]
        base = runs["all_on"]["probs"][:, keep]
        gated = runs[f"{name}_off"]["probs"][:, keep]
        equal = bool(np.array_equal(base, gated))
        dropped_nan = bool(np.isnan(runs[f"{name}_off"]["probs"][:, [j for j in range(len(classes)) if j not in keep]]).all())
        a_results[name] = equal
        report.append(f"| {name} off | {len(keep)} | {len(windows)} | {equal} | "
                      f"{np.max(np.abs(base - gated)):.3g} |")
        report.append(f"|  ↳ 꺼진 {name} 라벨이 결과에서 빠졌는가 | 4 | | {dropped_nan} | |")

    # ---- B ----
    report.append("\n## B. 가려진 후보 노출\n")
    report.append("2등 예측 = all_on run의 candidates에서 꺼진 detector 라벨을 뺀 첫 항목.\n")
    report.append("| 평가 폴더 | 끈 detector | 윈도우 | 원래 best_label | 게이트 후 best_label | "
                  "2등 예측과 일치 | best_label 바뀐 윈도우 |")
    report.append("|---|---|---|---|---|---|---|")
    b_rows = []
    for folder in sorted(set(folders)):
        idx = [i for i, f in enumerate(folders) if f == folder]
        owners = [FOLDER_OWNER[folder]] if folder in FOLDER_OWNER else list(DETECTORS)
        for owner in owners:
            off = set(owned[owner])
            orig = [runs["all_on"]["best"][i] for i in idx]
            gated = [runs[f"{owner}_off"]["best"][i] for i in idx]
            predicted = [
                next((c for c in runs["all_on"]["candidates"][i] if c not in off), None)
                for i in idx
            ]
            match = sum(p == g for p, g in zip(predicted, gated))
            changed = sum(o != g for o, g in zip(orig, gated))
            b_rows.append((folder, owner, len(idx), changed))
            report.append(f"| {folder} | {owner} | {len(idx)} | {dist(orig)} | {dist(gated)} | "
                          f"{match}/{len(idx)} | {changed} |")

    # 원래 best가 꺼진 detector 소속인 윈도우 중 다른 라벨이 노출된 비율 (전체 폴더)
    report.append("\n### 노출 요약 (전체 윈도우)\n")
    report.append("| 끈 detector | 원래 best가 그 detector 소속 | 그중 다른 라벨 노출 | 그중 (none) |")
    report.append("|---|---|---|---|")
    for name in DETECTORS:
        off = set(owned[name])
        hit = [i for i, b in enumerate(runs["all_on"]["best"]) if b in off]
        exposed = sum(runs[f"{name}_off"]["best"][i] is not None for i in hit)
        report.append(f"| {name} | {len(hit)} | {exposed} | {len(hit) - exposed} |")

    report.append("\n### 윈도우당 latency (in-process, ms)\n")
    report.append("| 설정 | mean | p95 |")
    report.append("|---|---|---|")
    for key, run in runs.items():
        lat = sorted(run["latency"][1:])
        report.append(f"| {key} | {statistics.mean(lat):.1f} | {lat[int(len(lat) * 0.95) - 1]:.1f} |")

    text = "\n".join(report)
    (out / "scores_report.md").write_text(text, encoding="utf-8")
    np.savez_compressed(
        out / "scores_raw.npz",
        classes=np.array(classes),
        folders=np.array(folders),
        paths=np.array(paths),
        **{f"{k}_probs": v["probs"] for k, v in runs.items()},
        **{f"{k}_best": np.array([b or "" for b in v["best"]]) for k, v in runs.items()},
    )
    print(text)
    return 0 if all(a_results.values()) else 1


# --------------------------------------------------------------------------- #
# C. cost
# --------------------------------------------------------------------------- #
def _rss_mb() -> float:
    import psutil

    return psutil.Process().memory_info().rss / (1024.0 * 1024.0)


def cmd_detector_cost(args: argparse.Namespace) -> int:
    """One detector in this (fresh) process: load time, latency, RSS delta as JSON."""
    import torch  # noqa: F401  공통 런타임은 baseline에 포함

    from detector_registry import build_detector, configure_torch_threads

    configure_torch_threads(args.torch_threads)
    rss0 = _rss_mb()
    started = time.perf_counter()
    detector = build_detector(args.name, batch_size=1)
    load_s = time.perf_counter() - started
    rss_loaded = _rss_mb()
    rng = np.random.default_rng(0)
    timings = []
    for i in range(args.warmup + args.iters):
        window = (rng.standard_normal(runtime_config.WINDOW_SAMPLES) * 0.05).astype(np.float32)
        t = time.perf_counter()
        detector.predict_proba_batch(window[None, :])
        if i >= args.warmup:
            timings.append((time.perf_counter() - t) * 1000.0)
    timings.sort()
    print("RESULT " + json.dumps({
        "name": args.name,
        "load_s": load_s,
        "mean_ms": statistics.mean(timings),
        "p95_ms": timings[int(len(timings) * 0.95) - 1],
        "rss_base_mb": rss0,
        "rss_load_mb": rss_loaded - rss0,
        "rss_after_infer_mb": _rss_mb() - rss0,
    }))
    return 0


def build_long_wav(data_root: Path, path: Path, n_clips: int) -> int:
    """Round-robin clips across folders into one long 16 kHz wav. Returns samples."""
    import soundfile as sf

    lists = [sorted(p.glob("*.wav")) for p in sorted(data_root.iterdir()) if p.is_dir()]
    picked = []
    i = 0
    while len(picked) < n_clips:
        group = lists[i % len(lists)]
        if i // len(lists) < len(group):
            picked.append(group[i // len(lists)])
        i += 1
    audio = np.concatenate([sf.read(p, dtype="float32")[0] for p in picked])
    sf.write(path, audio, runtime_config.SAMPLE_RATE)
    return len(audio)


def cmd_main_probe(main_args: list[str]) -> int:
    """Run main.main() in this process, timing every window from the inside.

    Window 1 is warmup and excluded. CPU = this process's user+sys time.
    """
    import psutil

    import audio_stream_controller
    import main as backend_main
    import parallel_inference

    me = psutil.Process()
    samples: list[tuple[float, float]] = []  # (cpu_s, wall_s) after each window
    latencies: list[float] = []
    peak = [0.0]

    original_window = audio_stream_controller.AudioStreamController.process_window
    original_predict = parallel_inference.ParallelInferenceEngine.predict

    def process_window(self, window):  # noqa: ANN001
        payload = original_window(self, window)
        samples.append((sum(me.cpu_times()[:2]), time.perf_counter()))
        peak[0] = max(peak[0], me.memory_info().rss / (1024.0 * 1024.0))
        return payload

    def predict(self, window):  # noqa: ANN001
        result = original_predict(self, window)
        latencies.append(result.total_latency_ms)
        return result

    audio_stream_controller.AudioStreamController.process_window = process_window
    parallel_inference.ParallelInferenceEngine.predict = predict
    code = backend_main.main(main_args)

    measured = len(samples) - 1
    cpu_s = samples[-1][0] - samples[0][0] if measured > 0 else 0.0
    wall_s = samples[-1][1] - samples[0][1] if measured > 0 else 0.0
    lat = latencies[1:]
    print("RESULT " + json.dumps({
        "windows": len(samples),
        "mean_latency_ms": statistics.mean(lat) if lat else 0.0,
        "cpu_s_per_window": cpu_s / measured if measured > 0 else 0.0,
        "cpu_percent": 100.0 * cpu_s / wall_s if wall_s else 0.0,
        "rss_mb": peak[0],
        "exit": code,
    }), flush=True)
    return code


def run_main(wav: Path, disabled: list[str], torch_threads: int) -> dict:
    """main.py --input-wav under one gate, in a fresh process (see cmd_main_probe)."""
    main_args = [
        "--no-ble", "--no-battery", "--no-skip-low-db",
        "--torch-threads", str(torch_threads),
        "--input-wav", str(wav), "--disabled-labels", ",".join(disabled),
    ]
    proc = subprocess.run(
        [sys.executable, __file__, "main-probe", *main_args],
        cwd=BACKEND_DIR, capture_output=True, text=True, encoding="utf-8",
        errors="replace", env=os.environ | {"PYTHONIOENCODING": "utf-8"},
    )
    line = next((l for l in proc.stdout.splitlines() if l.startswith("RESULT ")), None)
    if line is None:
        print(proc.stdout[-2000:], proc.stderr[-2000:], file=sys.stderr)
        raise SystemExit("main.py 실행 실패")
    return json.loads(line[len("RESULT "):])


def cmd_cost(args: argparse.Namespace) -> int:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    report = ["## C. 비용\n", "### C-1. detector별 (각각 새 프로세스, 랜덤 윈도우)\n",
              "| detector | load (s) | mean (ms) | p95 (ms) | RSS+ 로드 (MB) | RSS+ 추론 후 (MB) |",
              "|---|---|---|---|---|---|"]
    for name in DETECTORS:
        proc = subprocess.run(
            [sys.executable, __file__, "detector-cost", "--name", name,
             "--iters", str(args.iters), "--warmup", str(args.warmup),
             "--torch-threads", str(args.torch_threads)],
            cwd=BACKEND_DIR, capture_output=True, text=True, encoding="utf-8",
            errors="replace", env=os.environ | {"PYTHONIOENCODING": "utf-8"},
        )
        line = next((l for l in proc.stdout.splitlines() if l.startswith("RESULT ")), None)
        if line is None:
            print(proc.stdout[-2000:], proc.stderr[-2000:], file=sys.stderr)
            raise SystemExit(f"{name} 측정 실패")
        r = json.loads(line[len("RESULT "):])
        print(r, flush=True)
        report.append(f"| {name} | {r['load_s']:.1f} | {r['mean_ms']:.1f} | {r['p95_ms']:.1f} | "
                      f"{r['rss_load_mb']:.0f} | {r['rss_after_infer_mb']:.0f} |")
    report.append(f"\nbaseline(파이썬+numpy+torch import) RSS ≈ {r['rss_base_mb']:.0f} MB\n")

    # 라벨 목록은 checkpoint 기준과 같은 config 값을 쓴다(테스트로 일치 확인됨).
    owned = runtime_config.DETECTOR_CLASSES
    wav = out / "long_input.wav"
    samples = build_long_wav(Path(args.data_root), wav, args.clips)
    report.append(f"### C-2. main.py --input-wav ({samples / 16000:.0f}s, hop 1s, "
                  f"torch_threads={args.torch_threads})\n")
    report.append("1번 윈도우(warmup) 제외. 실시간 CPU% = CPU-s/윈도우 ÷ hop 1s (100% = 코어 1개).\n")
    report.append("| 설정 | 윈도우 | 평균 latency (ms) | CPU-s/윈도우 | 실시간 환산 CPU% | "
                  "--input-wav 중 CPU% | peak RSS (MB) |")
    report.append("|---|---|---|---|---|---|---|")
    configs = {
        "전부 켬": [],
        "yamnet 끔": owned["yamnet"],
        "efficientat 끔": owned["efficientat"],
        "mobilenetv4 끔": owned["mobilenetv4"],
        "전부 끔": [c for d in DETECTORS for c in owned[d]],
    }
    for key, disabled in configs.items():
        r = run_main(wav, disabled, args.torch_threads)
        print(key, r, flush=True)
        report.append(f"| {key} | {r['windows']} | {r['mean_latency_ms']:.1f} | "
                      f"{r['cpu_s_per_window']:.3f} | {100 * r['cpu_s_per_window']:.1f} | "
                      f"{r['cpu_percent']:.0f} | {r['rss_mb']:.0f} |")
    text = "\n".join(report)
    (out / "cost_report.md").write_text(text, encoding="utf-8")
    print(text)
    return 0


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if argv[:1] == ["main-probe"]:
        # main.py 인자를 그대로 넘기므로 argparse를 거치지 않는다.
        return cmd_main_probe(argv[1:])
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    threads = runtime_config.DEFAULT_TORCH_THREADS

    s = sub.add_parser("scores")
    s.add_argument("--data-root", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--limit-per-class", type=int, default=None)
    s.add_argument("--control-windows", type=int, default=300)
    s.add_argument("--torch-threads", type=int, default=threads)

    c = sub.add_parser("cost")
    c.add_argument("--data-root", required=True)
    c.add_argument("--out", required=True)
    c.add_argument("--clips", type=int, default=120)
    c.add_argument("--iters", type=int, default=30)
    c.add_argument("--warmup", type=int, default=3)
    c.add_argument("--torch-threads", type=int, default=threads)

    d = sub.add_parser("detector-cost")
    d.add_argument("--name", required=True, choices=DETECTORS)
    d.add_argument("--iters", type=int, default=30)
    d.add_argument("--warmup", type=int, default=3)
    d.add_argument("--torch-threads", type=int, default=threads)

    args = parser.parse_args(argv)
    return {
        "scores": cmd_scores,
        "cost": cmd_cost,
        "detector-cost": cmd_detector_cost,
    }[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
