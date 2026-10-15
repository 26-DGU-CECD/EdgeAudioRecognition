"""CPU-only preflight: open replay source audio before running GPU inference."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from project_io import write_csv, write_json


def audit_audio(manifest, reader):
    by_path = {}
    for e in manifest["episodes"]:
        by_path.setdefault(e["path"], []).append(e)
    rows = []
    for path, episodes in by_path.items():
        row = dict(path=path, labels="|".join(sorted({e["label"] for e in episodes})),
                   group_ids="|".join(sorted({e["group_id"] for e in episodes})),
                   phase_episodes=len(episodes), status="error", error=None,
                   sample_rate=None, channels=None, duration_s=None, shorter_than_2s=None,
                   decoded_prefix_samples=None, decoded_prefix_peak=None)
        try:
            info = reader.info(path)
            if info.samplerate <= 0 or info.frames <= 0 or info.channels <= 0:
                raise ValueError("empty audio or invalid sample rate/channels")
            prefix, sr = reader.read(path, frames=min(info.frames, round(2 * info.samplerate)),
                                     dtype="float32", always_2d=True)
            if sr != info.samplerate or prefix.size == 0 or not np.isfinite(prefix).all():
                raise ValueError("invalid decoded prefix")
            row.update(status="ok", sample_rate=int(info.samplerate), channels=int(info.channels),
                       duration_s=float(info.frames / info.samplerate),
                       shorter_than_2s=bool(info.frames < 2 * info.samplerate),
                       decoded_prefix_samples=len(prefix), decoded_prefix_peak=float(np.max(np.abs(prefix))))
        except Exception as exc:
            row["error"] = f"{type(exc).__name__}: {exc}"
        rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="selection_reference")
    args = parser.parse_args()
    out = Path(args.out).resolve()
    manifest_bytes = (out / "replay_manifest.json").read_bytes()
    manifest = json.loads(manifest_bytes)
    import soundfile as sf  # Existing audio environment; --help needs no codec dependency.
    rows = audit_audio(manifest, sf)
    passed = bool(rows) and all(r["status"] == "ok" for r in rows)
    summary = dict(passed=passed, unique_source_files=len(rows),
                   failed_files=sum(r["status"] != "ok" for r in rows),
                   short_files=sum(r["shorter_than_2s"] is True for r in rows),
                   manifest_sha256=hashlib.sha256(manifest_bytes).hexdigest(),
                   scope="Header and first 2s decoding check only. Does not verify label, event onset or whole-file integrity.")
    write_csv(out / "audio_preflight.csv", rows)
    write_json(out / "audio_preflight.json", summary)
    print(summary)
    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
