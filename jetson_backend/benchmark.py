#!/usr/bin/env python3
"""Measure detector latency against the configured realtime hop budget."""
from __future__ import annotations

import argparse
import os
import platform
import resource
import statistics
import time

import numpy as np

import runtime_config
from detector_registry import build_detector, configure_torch_threads
from parallel_inference import ParallelInferenceEngine


def percentile95(measurements: list[float]) -> float:
    if not measurements:
        raise ValueError("측정값이 없습니다.")
    ordered = sorted(float(value) for value in measurements)
    return ordered[max(0, int(len(ordered) * 0.95) - 1)]


def _rss_mb() -> float:
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return (
        rss / 1024.0
        if platform.system() == "Linux"
        else rss / (1024.0 * 1024.0)
    )


def _make_window(rng: np.random.Generator) -> np.ndarray:
    return (
        rng.standard_normal(runtime_config.WINDOW_SAMPLES).astype(np.float32)
        * np.float32(0.05)
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iters", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument(
        "--detectors",
        default=",".join(runtime_config.DEFAULT_DETECTORS),
    )
    parser.add_argument(
        "--hop-seconds",
        type=float,
        default=runtime_config.DEFAULT_HOP_SECONDS,
    )
    parser.add_argument(
        "--torch-threads",
        type=int,
        default=runtime_config.DEFAULT_TORCH_THREADS,
    )
    parser.add_argument("--concurrent", action="store_true")
    parser.add_argument("--checkpoints-dir", default=None)
    parser.add_argument("--efficientat-dir", default=None)
    args = parser.parse_args(argv)
    if args.iters <= 0 or args.warmup < 0 or args.hop_seconds <= 0:
        parser.error("--iters와 --hop-seconds는 양수, --warmup은 0 이상이어야 합니다.")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    runtime_config.apply_path_overrides(
        checkpoints_dir=args.checkpoints_dir,
        efficientat_dir=args.efficientat_dir,
    )
    names = [name.strip() for name in args.detectors.split(",") if name.strip()]
    unknown = [name for name in names if name not in runtime_config.DEFAULT_DETECTORS]
    if not names or unknown:
        raise SystemExit(f"알 수 없는 detector: {unknown}")

    windows = [
        _make_window(np.random.default_rng(index))
        for index in range(args.iters + args.warmup)
    ]
    threads = configure_torch_threads(args.torch_threads)
    print(
        f"platform={platform.machine()} cpus={os.cpu_count()} "
        f"torch_threads={threads} iters={args.iters} hop={args.hop_seconds}s"
    )
    print(f"baseline RSS: {_rss_mb():.0f} MB")

    detectors: dict[str, object] = {}
    rows: list[tuple[str, float, float, float, float]] = []
    for name in names:
        rss_before = _rss_mb()
        started = time.perf_counter()
        detector = build_detector(name, batch_size=1)
        load_seconds = time.perf_counter() - started
        rss_after = _rss_mb()
        detectors[name] = detector

        for window in windows[: args.warmup]:
            detector.predict_proba_batch(window[None, :])
        timings = []
        for window in windows[args.warmup :]:
            started = time.perf_counter()
            detector.predict_proba_batch(window[None, :])
            timings.append((time.perf_counter() - started) * 1000.0)

        mean_ms = statistics.mean(timings)
        p95_ms = percentile95(timings)
        rows.append((name, load_seconds, mean_ms, p95_ms, rss_after - rss_before))
        print(
            f"[{name}] load={load_seconds:.1f}s mean={mean_ms:.0f}ms "
            f"p95={p95_ms:.0f}ms RSS+{rss_after - rss_before:.0f}MB",
            flush=True,
        )

    total_p95 = sum(row[3] for row in rows)
    if args.concurrent and len(detectors) > 1:
        engine = ParallelInferenceEngine(detectors, concurrent=True)
        for window in windows[: args.warmup]:
            engine.predict(window)
        concurrent_timings = [
            engine.predict(window).total_latency_ms
            for window in windows[args.warmup :]
        ]
        engine.close()
        concurrent_p95 = percentile95(concurrent_timings)
        total_p95 = min(total_p95, concurrent_p95)
        print(
            f"[concurrent] mean={statistics.mean(concurrent_timings):.0f}ms "
            f"p95={concurrent_p95:.0f}ms"
        )

    budget_ms = args.hop_seconds * 1000.0
    print(f"hop budget={budget_ms:.0f}ms, measured p95={total_p95:.0f}ms")
    if total_p95 <= budget_ms:
        print("REALTIME OK")
        return 0
    print(
        "NEEDS ATTENTION: --hop-seconds 2.0, --concurrent, "
        "또는 --detectors 축소를 순서대로 검토하세요."
    )
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
