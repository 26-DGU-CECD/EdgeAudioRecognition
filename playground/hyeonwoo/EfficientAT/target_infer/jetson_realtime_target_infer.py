#!/usr/bin/env python3
"""Realtime microphone inference for EfficientAT target labels on Jetson.

The script records short ALSA chunks with arecord, skips quiet chunks, and runs
the same target-label aggregation used by efficientat_target_infer.py.

Example:
    python3 -u jetson_realtime_target_infer.py

For a 6-channel ReSpeaker-style mic array:
    python3 -u jetson_realtime_target_infer.py \
      --device auto --channels 6 --channel-index 0
"""

from __future__ import annotations

import argparse
import re
import subprocess
import time
import wave
from pathlib import Path

import numpy as np
import torch

import efficientat_target_infer as target_infer


DBFS_FLOOR = -120.0
DEFAULT_DB_OFFSET = 80.0
ARECORD_LIST_TIMEOUT = 3.0


ARECORD_DEVICE_RE = re.compile(
    r"card\s+(?P<card>\d+):\s+(?P<card_name>[^[]+)\[[^\]]+\],\s+"
    r"device\s+(?P<device>\d+):\s+(?P<device_name>.+)"
)


class NullContext:
    def __enter__(self):
        return None

    def __exit__(self, *args):
        return False


def amp_context(use_cuda: bool, use_amp: bool):
    if use_cuda and use_amp and hasattr(torch.cuda, "amp"):
        return torch.cuda.amp.autocast()
    return NullContext()


def read_wav_select_channel(path: Path, channel_index: int) -> tuple[np.ndarray, int, int]:
    with wave.open(str(path), "rb") as wf:
        sample_rate = wf.getframerate()
        channels = wf.getnchannels()
        sample_width = wf.getsampwidth()
        nframes = wf.getnframes()
        raw = wf.readframes(nframes)

    if sample_width == 2:
        audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    elif sample_width == 4:
        audio = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0
    else:
        raise RuntimeError("Only 16-bit or 32-bit PCM WAV is supported.")

    if channels > 1:
        audio = audio.reshape(-1, channels)
        if channel_index == -1:
            audio = audio.mean(axis=1)
        else:
            if channel_index < 0 or channel_index >= channels:
                raise RuntimeError(f"channel-index {channel_index} is out of range for {channels} channels")
            audio = audio[:, channel_index]

    return audio.astype(np.float32), sample_rate, channels


def resample_linear(audio: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    if src_sr == dst_sr:
        return audio.astype(np.float32)
    if len(audio) == 0:
        return audio.astype(np.float32)

    duration = len(audio) / float(src_sr)
    old_x = np.linspace(0.0, duration, num=len(audio), endpoint=False)
    new_len = int(round(duration * dst_sr))
    new_x = np.linspace(0.0, duration, num=new_len, endpoint=False)
    return np.interp(new_x, old_x, audio).astype(np.float32)


def calc_dbfs(audio: np.ndarray) -> float:
    if audio is None or len(audio) == 0:
        return DBFS_FLOOR

    samples = audio.astype(np.float64, copy=False)
    rms = float(np.sqrt(np.mean(samples * samples)))
    if rms <= 1e-12:
        return DBFS_FLOOR
    return float(20.0 * np.log10(rms))


def dbfs_to_display_db(dbfs: float, offset: float) -> float:
    display_db = float(dbfs) + float(offset)
    return max(0.0, display_db)


def list_capture_devices() -> list[dict[str, str]]:
    try:
        result = subprocess.run(
            ["arecord", "-l"],
            check=False,
            text=True,
            capture_output=True,
            timeout=ARECORD_LIST_TIMEOUT,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise RuntimeError(f"Failed to run arecord -l: {exc}") from exc

    output = "\n".join(part for part in [result.stdout, result.stderr] if part)
    devices = []
    for line in output.splitlines():
        match = ARECORD_DEVICE_RE.search(line)
        if not match:
            continue
        card = match.group("card")
        device = match.group("device")
        devices.append(
            {
                "alsa": f"plughw:{card},{device}",
                "card": card,
                "device": device,
                "card_name": match.group("card_name").strip(),
                "device_name": match.group("device_name").strip(),
            }
        )
    return devices


def format_capture_devices(devices: list[dict[str, str]]) -> str:
    if not devices:
        return "  none"
    return "\n".join(
        "  {alsa}  card={card} device={device}  {card_name} / {device_name}".format(**device)
        for device in devices
    )


def device_requested_card_device(device_name: str) -> tuple[str, str] | None:
    match = re.fullmatch(r"(?:plug)?hw:(\d+),(\d+)", device_name)
    if not match:
        return None
    return match.group(1), match.group(2)


def resolve_recording_device(requested: str) -> str:
    devices = list_capture_devices()
    if requested != "auto":
        requested_card_device = device_requested_card_device(requested)
        if requested_card_device is None:
            return requested
        card, device = requested_card_device
        if any(item["card"] == card and item["device"] == device for item in devices):
            return requested
        if not devices:
            raise RuntimeError(
                f"Requested audio device {requested!r} was not found, and arecord -l returned no capture devices."
            )
        fallback = devices[0]["alsa"]
        print(
            f"Warning: requested audio device {requested!r} was not found. "
            f"Using {fallback!r} instead.\nAvailable capture devices:\n{format_capture_devices(devices)}",
            flush=True,
        )
        return fallback

    if not devices:
        raise RuntimeError("No ALSA capture devices found by arecord -l.")

    preferred_terms = ("usb", "respeaker", "mic", "microphone", "array")
    for item in devices:
        description = f"{item['card_name']} {item['device_name']}".lower()
        if any(term in description for term in preferred_terms):
            return item["alsa"]
    return devices[0]["alsa"]


def prepare_waveform(audio: np.ndarray, sample_rate: int, duration_sec: float | None) -> np.ndarray:
    audio = np.asarray(audio, dtype=np.float32)
    if duration_sec is None or duration_sec <= 0:
        return audio

    target_len = int(round(float(duration_sec) * int(sample_rate)))
    if target_len <= 0:
        raise RuntimeError("duration-sec must be positive")

    if audio.size < target_len:
        return np.pad(audio, (0, target_len - audio.size), mode="constant").astype(np.float32)
    if audio.size > target_len:
        return audio[-target_len:].astype(np.float32)
    return audio.astype(np.float32)


def record_chunk(args: argparse.Namespace, wav_path: Path) -> int:
    cmd = [
        "arecord",
        "-q",
        "-D",
        args.device,
        "-f",
        "S16_LE",
        "-r",
        str(args.rate),
        "-c",
        str(args.channels),
        "-d",
        str(args.seconds),
        str(wav_path),
    ]
    return subprocess.call(cmd)


def predict_targets(
    model: torch.nn.Module,
    mel: torch.nn.Module,
    audio: np.ndarray,
    device: torch.device,
    target_indexes: dict[str, list[int]],
    audioset_labels: list[str],
    args: argparse.Namespace,
) -> tuple[list[tuple[str, dict]], float]:
    waveform = torch.from_numpy(audio[None, :]).to(device)
    use_cuda = device.type == "cuda"

    if use_cuda:
        torch.cuda.synchronize()
    start = time.time()

    with torch.no_grad(), amp_context(use_cuda, args.amp):
        spec = mel(waveform)
        output = model(spec.unsqueeze(0))
        logits = output[0] if isinstance(output, (tuple, list)) else output
        probs = torch.sigmoid(logits.float()).squeeze().detach().cpu().numpy()

    if use_cuda:
        torch.cuda.synchronize()
    infer_time = time.time() - start

    targets = target_infer.aggregate_targets(probs, target_indexes, audioset_labels, args.aggregate)
    ordered = sorted(targets.items(), key=lambda item: item[1]["score"], reverse=True)
    return ordered, infer_time


def format_predictions(ordered: list[tuple[str, dict]], topk: int, threshold: float) -> str:
    shown = []
    for target_label, payload in ordered[:topk]:
        score = float(payload["score"])
        if score < threshold:
            continue
        shown.append(
            f"{target_label} {score:.3f}"
            f"({payload['best_audioset_label']}={payload['best_audioset_score']:.3f})"
        )

    if shown:
        return " / ".join(shown)

    target_label, payload = ordered[0]
    return f"below-threshold top={target_label} {float(payload['score']):.3f}"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run realtime Jetson microphone inference for target labels.")
    parser.add_argument("--device", default="auto", help="ALSA input device, e.g. auto, default, or plughw:1,0")
    parser.add_argument("--list-devices", action="store_true", help="Print ALSA capture devices and exit.")
    parser.add_argument("--efficientat-dir", type=Path, default=None, help="EfficientAT source dir. Default: auto-detect.")
    parser.add_argument("--target-mapping", type=Path, default=None, help="Optional JSON: target_label -> AudioSet labels.")

    parser.add_argument("--model-name", default="mn10_as")
    parser.add_argument("--strides", nargs=4, default=[2, 2, 2, 2], type=int)
    parser.add_argument("--head-type", default="mlp")
    parser.add_argument("--ensemble", nargs="+", default=[])

    parser.add_argument("--rate", type=int, default=16000, help="Microphone recording sample rate.")
    parser.add_argument("--channels", type=int, default=1, help="Microphone channel count passed to arecord.")
    parser.add_argument("--channel-index", type=int, default=0, help="0=first channel, -1=average all channels.")
    parser.add_argument("--seconds", type=int, default=2, help="Recording seconds per realtime decision.")

    parser.add_argument("--sample-rate", type=int, default=32000, help="EfficientAT model sample rate.")
    parser.add_argument("--duration-sec", type=float, default=2.0, help="Model input length after pad/crop.")
    parser.add_argument("--window-size", type=int, default=800)
    parser.add_argument("--hop-size", type=int, default=320)
    parser.add_argument("--n-mels", type=int, default=128)

    parser.add_argument("--min-db", type=float, default=30.0, help="Skip chunks with display dB <= this value.")
    parser.add_argument("--db-offset", type=float, default=DEFAULT_DB_OFFSET, help="display_dB = dBFS + offset.")
    parser.add_argument("--aggregate", choices=["max", "mean"], default="max")
    parser.add_argument("--topk", type=int, default=3)
    parser.add_argument("--threshold", type=float, default=0.0, help="Only print labels with target score >= threshold.")
    parser.add_argument("--tmp-wav", type=Path, default=Path("/tmp/efficientat_target_live.wav"))
    parser.add_argument("--print-skipped", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--amp", action="store_true", help="Enable CUDA AMP. Leave off if Jetson is unstable.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.tmp_wav = args.tmp_wav.expanduser().resolve()
    args.tmp_wav.parent.mkdir(parents=True, exist_ok=True)
    if args.target_mapping is not None:
        args.target_mapping = args.target_mapping.expanduser().resolve()

    if args.list_devices:
        print(format_capture_devices(list_capture_devices()))
        return

    args.device = resolve_recording_device(args.device)

    use_cuda = (not args.cpu) and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    if args.efficientat_dir is not None:
        target_infer.EFFICIENTAT_ROOT = args.efficientat_dir.expanduser().resolve()

    target_mapping = target_infer.load_target_mapping(args.target_mapping)
    model, mel, audioset_labels = target_infer.load_model(args, device)
    target_indexes = target_infer.build_label_indexes(target_mapping, audioset_labels)

    print("device:", device)
    print("EfficientAT:", target_infer.EFFICIENTAT_ROOT)
    print("model:", args.model_name)
    print("audio device:", args.device)
    print(f"record: {args.rate}Hz, {args.channels}ch, {args.seconds}s, channel-index={args.channel_index}")
    print(f"model input: {args.sample_rate}Hz, {args.duration_sec:.2f}s")
    print(f"min-db: {args.min_db:.1f} dB; display_dB = dBFS + {args.db_offset:.1f}")
    print("Ctrl+C to stop")
    print("-" * 60)

    while True:
        loop_start = time.time()
        rc = record_chunk(args, args.tmp_wav)
        if rc != 0:
            print("arecord failed. Check --device, --rate, and --channels.", flush=True)
            time.sleep(1.0)
            continue

        audio, src_sr, detected_channels = read_wav_select_channel(args.tmp_wav, args.channel_index)
        audio = resample_linear(audio, src_sr, args.sample_rate)

        dbfs = calc_dbfs(audio)
        display_db = dbfs_to_display_db(dbfs, args.db_offset)
        if display_db <= args.min_db:
            if args.print_skipped:
                total_time = time.time() - loop_start
                print(
                    f"[{time.strftime('%H:%M:%S')}] total={total_time:.3f}s "
                    f"db={display_db:.1f}dB <= {args.min_db:.1f}dB | skipped",
                    flush=True,
                )
            continue

        waveform = prepare_waveform(audio, args.sample_rate, args.duration_sec)
        ordered, infer_time = predict_targets(
            model,
            mel,
            waveform,
            device,
            target_indexes,
            audioset_labels,
            args,
        )
        total_time = time.time() - loop_start
        prediction_text = format_predictions(ordered, args.topk, args.threshold)

        print(
            f"[{time.strftime('%H:%M:%S')}] infer={infer_time:.3f}s total={total_time:.3f}s "
            f"db={display_db:.1f}dB ch={detected_channels} | {prediction_text}",
            flush=True,
        )


if __name__ == "__main__":
    main()
