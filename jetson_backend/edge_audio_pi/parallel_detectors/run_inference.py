"""
Run one detector over the manifest and write per-clip sigmoid scores.

The detector is scored on ALL clips it needs (its own positives + every other clip
as a negative) so the evaluator can compute per-class binary precision/recall/PR-AUC.

  * efficientat / yamnet (zero-shot) -> default: score the whole dataset ("all")
  * mobilenetv4    (trained)         -> default: score the "test" split only
                                        (never evaluate on its own training data)

Output: parallel_detectors/scores/scores_<detector>.csv with columns
    filepath, true_label, prob_<class1>, prob_<class2>, ...
Resumable: already-scored filepaths are skipped, so the run can be stopped/resumed.

Usage (Env A):
    python parallel_detectors/run_inference.py --detector efficientat
    python parallel_detectors/run_inference.py --detector mobilenetv4          # -> test split
Usage (Env B, TF):
    python parallel_detectors/run_inference.py --detector yamnet
"""
from __future__ import annotations

import argparse
import os
import sys

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import numpy as np
import pandas as pd
import soundfile as sf
from tqdm import tqdm

from parallel_detectors import config


def load_waveform(path: str) -> np.ndarray:
    """Load a clip as float32 mono @16 kHz, fixed to CLIP_SAMPLES length."""
    wav, sr = sf.read(path, dtype="float32", always_2d=False)
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    if sr != config.DATASET_SR:
        import librosa
        wav = librosa.resample(wav, orig_sr=sr, target_sr=config.DATASET_SR)
    n = config.CLIP_SAMPLES
    if len(wav) < n:
        wav = np.pad(wav, (0, n - len(wav)))
    elif len(wav) > n:
        wav = wav[:n]
    return wav.astype(np.float32)


def _resolve_ckpt(name: str, variant: str):
    """Return a checkpoint path to use, or None for zero-shot.

    variant: 'auto' -> trained if the checkpoint exists, else zero-shot.
             'trained' -> require the checkpoint. 'zeroshot' -> ignore it.
    """
    ckpt = {"efficientat": config.EFFICIENTAT_CKPT,
            "yamnet": config.YAMNET_CKPT,
            "mobilenetv4": config.MOBILENETV4_CKPT}[name]
    if variant == "zeroshot":
        return None
    if variant == "trained":
        if not ckpt.exists():
            raise FileNotFoundError(f"{name}: trained checkpoint missing: {ckpt}")
        return ckpt
    return ckpt if ckpt.exists() else None  # auto


def build_detector(name: str, batch_size: int, ckpt):
    if name == "efficientat":
        from parallel_detectors.detectors.efficientat_detector import EfficientATDetector
        return EfficientATDetector(batch_size=batch_size, ckpt=ckpt)
    if name == "yamnet":
        from parallel_detectors.detectors.yamnet_detector import YAMNetDetector
        return YAMNetDetector(ckpt=ckpt)
    if name == "mobilenetv4":
        from parallel_detectors.detectors.mobilenetv4_detector import MobileNetV4Detector
        return MobileNetV4Detector(ckpt_path=ckpt, batch_size=batch_size)
    raise ValueError(f"Unknown detector: {name}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--detector", required=True,
                    choices=["efficientat", "yamnet", "mobilenetv4"])
    ap.add_argument("--split", default=None,
                    choices=["all", "train", "val", "test"],
                    help="Default: 'test' for trained detectors, 'all' for zero-shot.")
    ap.add_argument("--variant", default="auto", choices=["auto", "trained", "zeroshot"],
                    help="auto: trained if a checkpoint exists, else zero-shot.")
    ap.add_argument("--manifest", default=str(config.MANIFEST_CSV))
    ap.add_argument("--batch-size", type=int, default=64,
                    help="Torch sub-batch size inside the detector.")
    ap.add_argument("--chunk", type=int, default=256,
                    help="Rows loaded from disk per inference call / flush.")
    ap.add_argument("--limit", type=int, default=None,
                    help="Only score the first N (post-filter) rows (debug).")
    args = ap.parse_args()

    # Resolve mode first — it drives the default eval split. A TRAINED detector must
    # be evaluated on the held-out test split (scoring 'all' would leak its training
    # data); a zero-shot detector has no training data, so it scores the full set.
    ckpt = _resolve_ckpt(args.detector, args.variant)
    if args.detector == "mobilenetv4" and ckpt is None:
        ckpt = config.MOBILENETV4_CKPT  # always trained
    mode = "trained" if ckpt else "zeroshot"
    split = args.split or ("test" if mode == "trained" else "all")

    man = pd.read_csv(args.manifest)
    if split != "all":
        man = man[man["split"] == split]
    man = man.reset_index(drop=True)
    if args.limit is not None:
        man = man.iloc[:args.limit]
    print(f"[run] detector={args.detector} mode={mode} split={split} rows={len(man):,}"
          + (f" ckpt={ckpt}" if ckpt else ""))

    config.SCORES_DIR.mkdir(parents=True, exist_ok=True)
    out_csv = config.SCORES_DIR / f"scores_{args.detector}.csv"

    # Resume: skip filepaths already scored. Guard against mixing modes in one file.
    done: set[str] = set()
    if out_csv.exists():
        prev = pd.read_csv(out_csv, usecols=lambda c: c in ("filepath", "mode"))
        prev_modes = set(prev["mode"].unique()) if "mode" in prev.columns else {"?"}
        if prev_modes - {mode}:
            raise SystemExit(
                f"[run] {out_csv} already holds mode(s) {prev_modes}, but this run is "
                f"'{mode}'. Delete the file to re-score:\n  rm {out_csv}")
        done = set(prev["filepath"])
        print(f"[run] resuming; {len(done):,} clips already scored.")
    todo = man[~man["filepath"].isin(done)].reset_index(drop=True)
    if todo.empty:
        print("[run] nothing to do.")
        return

    detector = build_detector(args.detector, args.batch_size, ckpt)
    prob_cols = [f"prob_{c}" for c in detector.target_classes]

    header_written = out_csv.exists()
    for start in tqdm(range(0, len(todo), args.chunk), desc=args.detector):
        rows = todo.iloc[start:start + args.chunk]
        wavs = np.stack([load_waveform(p) for p in rows["filepath"]])
        probs = detector.predict_proba_batch(wavs)  # [n, n_classes]
        out = pd.DataFrame({"filepath": rows["filepath"].values,
                            "true_label": rows["folder"].values,
                            "mode": mode})
        for j, col in enumerate(prob_cols):
            out[col] = probs[:, j]
        out.to_csv(out_csv, mode="a", header=not header_written, index=False)
        header_written = True

    print(f"[run] done -> {out_csv}")


if __name__ == "__main__":
    main()
